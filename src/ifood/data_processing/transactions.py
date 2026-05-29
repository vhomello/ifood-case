from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

def read_transactions_json(spark: SparkSession, base_path: Path | str, name: str = "transactions.json") -> DataFrame:
    file_path = Path(base_path) / name
    return spark.read.json(str(file_path))


def get_transaction_id(df: DataFrame) -> DataFrame:
    return df.withColumn("transaction_id", F.monotonically_increasing_id())


def read_transactions_array(df: DataFrame) -> DataFrame:
    return (
        df.withColumn(
            "offer_id", F.coalesce(F.col("value.offer id"), F.col("value.offer_id"))
        )
        .withColumn("transaction_amount", F.col("value.amount"))
        .withColumn("offer_reward", F.col("value.reward"))
        .drop("value")
    )
