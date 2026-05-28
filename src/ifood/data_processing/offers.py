from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


def read_offer_json(
    spark: SparkSession, base_path: Path | str, name: str = "offers.json"
) -> DataFrame:
    file_path = Path(base_path) / name
    return spark.read.json(str(file_path)).withColumnRenamed("id", "offer_id")


def get_offer_channels(df: DataFrame) -> DataFrame:
    def extract_channel(x):
        return F.array_contains(F.col("channels"), x).cast("int")

    return (
        df.withColumn("web_channel", extract_channel("web"))
        .withColumn("mobile_channel", extract_channel("mobile"))
        .withColumn("email_channel", extract_channel("email"))
        .withColumn("social_channel", extract_channel("social"))
        .drop("channels")
    )
