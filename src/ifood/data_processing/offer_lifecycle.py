from pyspark.sql import functions as F
from pyspark.sql import DataFrame
import pandas as pd


def _cast_id_columns_safely(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    for col in id_cols:
        df[col] = df[col].astype(str).replace({"nan": None, "None": None})
    return df


def _cast_numeric_columns_safely(
    df: pd.DataFrame, numeric_cols: list[str]
) -> pd.DataFrame:
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _sort_timeline_with_priority(pdf: pd.DataFrame, priority_map: dict) -> pd.DataFrame:
    pdf["priority"] = pdf["event_type"].map(priority_map)
    return pdf.sort_values(by=["time", "priority"]).reset_index(drop=True)


def _pop_latest_active_match(active_items: list, event_time: float) -> dict | None:
    for i in range(len(active_items) - 1, -1, -1):
        item = active_items[i]
        if item["received_at"] <= event_time <= item["ended_at"]:
            return active_items.pop(i)
    return None


def _match_views_to_receives(pdf: pd.DataFrame) -> pd.DataFrame:
    VIEW_SCHEMA_COLS = [
        "account_id",
        "offer_id",
        "transaction_id",
        "received_at",
        "ended_at",
        "view_transaction_id",
        "viewed_at",
    ]
    ID_COLS = ["transaction_id", "view_transaction_id"]
    NUMERIC_COLS = ["received_at", "ended_at", "viewed_at"]

    pdf = _sort_timeline_with_priority(pdf, {"received": 0, "viewed": 1})

    active_receives = []
    matched_rows = []

    for _, row in pdf.iterrows():
        if row["event_type"] == "received":
            active_receives.append(
                {
                    "tx_id": row["tx_id"],
                    "received_at": row["time"],
                    "ended_at": row["ended_at"],
                }
            )

        elif row["event_type"] == "viewed":
            matched = _pop_latest_active_match(active_receives, row["time"])
            if matched:
                matched_rows.append(
                    {
                        "account_id": row["account_id"],
                        "offer_id": row["offer_id"],
                        "transaction_id": matched["tx_id"],
                        "received_at": matched["received_at"],
                        "ended_at": matched["ended_at"],
                        "view_transaction_id": row["tx_id"],
                        "viewed_at": row["time"],
                    }
                )

    for unviewed in active_receives:
        matched_rows.append(
            {
                "account_id": pdf["account_id"].iloc[0],
                "offer_id": pdf["offer_id"].iloc[0],
                "transaction_id": unviewed["tx_id"],
                "received_at": unviewed["received_at"],
                "ended_at": unviewed["ended_at"],
                "view_transaction_id": None,
                "viewed_at": None,
            }
        )

    if not matched_rows:
        return pd.DataFrame(columns=VIEW_SCHEMA_COLS)

    out = pd.DataFrame(matched_rows)
    out = _cast_id_columns_safely(out, ID_COLS)
    out = _cast_numeric_columns_safely(out, NUMERIC_COLS)
    return out


def _match_completions_to_history(pdf: pd.DataFrame) -> pd.DataFrame:
    COMP_SCHEMA_COLS = [
        "account_id",
        "offer_id",
        "transaction_id",
        "received_at",
        "ended_at",
        "view_transaction_id",
        "viewed_at",
        "comp_transaction_id",
        "completed_at",
    ]
    ID_COLS = ["transaction_id", "view_transaction_id", "comp_transaction_id"]
    NUMERIC_COLS = ["received_at", "ended_at", "viewed_at", "completed_at"]

    pdf = _sort_timeline_with_priority(pdf, {"history": 0, "completed": 1})

    active_histories = []
    matched_rows = []

    for _, row in pdf.iterrows():
        if row["event_type"] == "history":
            active_histories.append(
                {
                    "tx_id": row["tx_id"],
                    "received_at": row["time"],
                    "ended_at": row["ended_at"],
                    "view_transaction_id": row["view_transaction_id"],
                    "viewed_at": row["viewed_at"],
                }
            )

        elif row["event_type"] == "completed":
            matched = _pop_latest_active_match(active_histories, row["time"])
            if matched:
                matched_rows.append(
                    {
                        "account_id": row["account_id"],
                        "offer_id": row["offer_id"],
                        "transaction_id": matched["tx_id"],
                        "received_at": matched["received_at"],
                        "ended_at": matched["ended_at"],
                        "view_transaction_id": matched["view_transaction_id"],
                        "viewed_at": matched["viewed_at"],
                        "comp_transaction_id": row["tx_id"],
                        "completed_at": row["time"],
                    }
                )

    for uncompleted in active_histories:
        matched_rows.append(
            {
                "account_id": pdf["account_id"].iloc[0],
                "offer_id": pdf["offer_id"].iloc[0],
                "transaction_id": uncompleted["tx_id"],
                "received_at": uncompleted["received_at"],
                "ended_at": uncompleted["ended_at"],
                "view_transaction_id": uncompleted["view_transaction_id"],
                "viewed_at": uncompleted["viewed_at"],
                "comp_transaction_id": None,
                "completed_at": None,
            }
        )

    if not matched_rows:
        return pd.DataFrame(columns=COMP_SCHEMA_COLS)

    out = pd.DataFrame(matched_rows)
    out = _cast_id_columns_safely(out, ID_COLS)
    out = _cast_numeric_columns_safely(out, NUMERIC_COLS)
    return out


def get_received_events(transactions: DataFrame) -> DataFrame:
    return transactions.where(F.col("event") == "offer received").select(
        F.col("transaction_id"),
        F.col("account_id"),
        F.col("offer_id"),
        F.col("time_since_test_start").alias("received_at"),
    )


def get_offer_end(received: DataFrame, offers: DataFrame) -> DataFrame:
    offer_durations = offers.select("offer_id", "duration")
    return (
        received.join(offer_durations, "offer_id", "inner")
        .withColumn("ended_at", F.col("received_at") + F.col("duration"))
        .drop("duration")
    )


def get_views(received_with_end: DataFrame, transactions: DataFrame) -> DataFrame:
    raw_views = transactions.where(F.col("event") == "offer viewed").select(
        F.col("account_id"),
        F.col("offer_id"),
        F.col("transaction_id").alias("tx_id"),
        F.col("time_since_test_start").alias("time"),
        F.lit("viewed").alias("event_type"),
        F.lit(None).cast("double").alias("ended_at"),
    )

    receives_as_timeline_events = received_with_end.select(
        "account_id",
        "offer_id",
        F.col("transaction_id").alias("tx_id"),
        F.col("received_at").alias("time"),
        F.lit("received").alias("event_type"),
        F.col("ended_at"),
    )

    view_output_schema = """
        account_id string, offer_id string,
        transaction_id string, received_at double, ended_at double,
        view_transaction_id string, viewed_at double
    """

    return (
        raw_views.unionByName(receives_as_timeline_events)
        .groupBy("account_id", "offer_id")
        .applyInPandas(_match_views_to_receives, schema=view_output_schema)
    )


def get_complete(views: DataFrame, transactions: DataFrame) -> DataFrame:
    raw_completions = transactions.where(F.col("event") == "offer completed").select(
        F.col("account_id"),
        F.col("offer_id"),
        F.col("transaction_id").alias("tx_id"),
        F.col("time_since_test_start").alias("time"),
        F.lit("completed").alias("event_type"),
        F.lit(None).cast("double").alias("ended_at"),
        F.lit(None).cast("string").alias("view_transaction_id"),
        F.lit(None).cast("double").alias("viewed_at"),
    )

    views_as_timeline_history = views.select(
        "account_id",
        "offer_id",
        F.col("transaction_id").alias("tx_id"),
        F.col("received_at").alias("time"),
        F.lit("history").alias("event_type"),
        F.col("ended_at"),
        F.col("view_transaction_id"),
        F.col("viewed_at"),
    )

    completion_output_schema = """
        account_id string, offer_id string,
        transaction_id string, received_at double, ended_at double,
        view_transaction_id string, viewed_at double,
        comp_transaction_id string, completed_at double
    """

    return (
        views_as_timeline_history.unionByName(raw_completions)
        .groupBy("account_id", "offer_id")
        .applyInPandas(_match_completions_to_history, schema=completion_output_schema)
    )


def cast_ids(df: DataFrame) -> DataFrame:
    return (
        df.withColumn(
            "transaction_id", F.col("transaction_id").cast("float").cast("bigint")
        )
        .withColumn(
            "view_transaction_id",
            F.col("view_transaction_id").cast("float").cast("bigint"),
        )
        .withColumn(
            "comp_transaction_id",
            F.col("comp_transaction_id").cast("float").cast("bigint"),
        )
    )


def get_overlap_count(df: DataFrame) -> DataFrame:
    lifecycle_period = df.withColumn(
        "period_end", F.least(F.col("ended_at"), F.col("completed_at"))
    )

    overlap_count = (
        lifecycle_period.alias("a")
        .join(
            lifecycle_period.alias("b"),
            (F.col("a.account_id") == F.col("b.account_id"))
            & (F.col("a.received_at") < F.col("b.period_end"))
            & (F.col("a.period_end") > F.col("b.received_at")),
            "left",
        )
        .where(F.col("a.transaction_id") != F.col("b.transaction_id"))
        .groupBy("a.account_id", "a.transaction_id")
        .agg(F.count("b.offer_id").alias("n_overlapping_offers"))
    )

    return df.join(overlap_count, ["account_id", "transaction_id"], "left").withColumn(
        "n_overlapping_offers", F.coalesce(F.col("n_overlapping_offers"), F.lit(0))
    )


def build_offer_lifecycle(transactions: DataFrame, offers: DataFrame) -> DataFrame:
    return (
        get_received_events(transactions)
        .transform(get_offer_end, offers)
        .transform(get_views, transactions)
        .transform(get_complete, transactions)
        .transform(cast_ids)
        .transform(get_overlap_count)
    )
