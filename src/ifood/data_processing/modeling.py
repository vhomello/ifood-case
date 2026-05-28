from pyspark.sql import functions as F
from pyspark.sql.window import Window


def select_offer(df):
    w = Window.partitionBy("account_id").orderBy("received_at")

    with_days_between = (
        df.withColumn("next_viewed_at", F.lead("viewed_at", 1).over(w))
        .withColumn(
            "offer_ended_at",
            F.least(F.col("ended_at"), F.col("completed_at"), F.lit(29)),
        )
        .withColumn(
            "days_between",
            F.coalesce(F.col("next_viewed_at"), F.lit(29)) - F.col("offer_ended_at"),
        )
    )

    logic = (
        (F.col("viewed_at").isNotNull())
        & (F.col("n_overlapping_offers") == 0)
        & (F.col("days_between") >= 4)
    )

    return with_days_between.where(logic).select(
        F.col("transaction_id"), F.lit(True).alias("is_offer"), F.col("offer_ended_at")
    )


def select_non_offer(transactions):
    first_4_days = transactions.where(F.col("time_since_test_start") < 5.0)

    account_with_offer = (
        first_4_days.where(F.col("event").contains("offer"))
        .select("account_id")
        .distinct()
    )

    return (
        first_4_days.join(account_with_offer, on="account_id", how="left_anti")
        .where(F.col("time_since_test_start") == 0)
        .select(
            F.col("transaction_id"),
            F.lit(False).alias("is_offer"),
            F.lit(0).alias("offer_ended_at"),
        )
    )


def get_rows(selected_offer, non_offer):
    return selected_offer.unionByName(non_offer)


def get_target(df, transactions):
    return (
        df.alias("l")
        .withColumn("censorship_end", F.col("offer_ended_at") + F.lit(4))
        .join(
            transactions.alias("r")
            .where(F.col("event") == "transaction")
            .select("account_id", "time_since_test_start", "transaction_amount"),
            (F.col("l.account_id") == F.col("r.account_id"))
            & (F.col("l.offer_ended_at") <= F.col("r.time_since_test_start"))
            & (F.col("l.censorship_end") >= F.col("r.time_since_test_start")),
        )
        .drop(F.col("r.account_id"), F.col("r.time_since_test_start"))
        .groupBy("transaction_id")
        .agg(
            F.sum("transaction_amount").alias("target"),
        )
    )


def join_all_dataframes(rows, target, transactions, profile, offers):
    transactions_cols = [
        "transaction_id",
        "offer_id",
        "account_idtime_since_test_start",
    ]

    customer_cols = ["account_id", "age", "credit_card_limit", "gender", "account_age"]

    offer_cols = [
        "offer_id",
        "discount_value",
        "duration",
        "min_value",
        "offer_type",
        "web_channel",
        "mobile_channel",
        "email_channel",
        "social_channel",
    ]

    return (
        rows.join(target, "transaction_id", "inner")
        .join(transactions.select(transactions_cols), on="transaction_id", how="inner")
        .join(profile.select(customer_cols), on="account_id", how="inner")
        .join(offers.select(offer_cols), on="offer_id", how="inner")
    )


def get_account_split(df):
    split_col = F.when(
        (F.abs(F.hash(F.col("account_id"))) % 100) < 70, F.lit("train")
    ).otherwise(F.lit("test"))

    return df.withColumn("split", split_col)
