from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType, DateType

def read_profile_json(spark: SparkSession, base_path: Path | str, name: str = "profile.json") -> DataFrame:
    profile_schema = StructType(
        [
            StructField("age", LongType(), True),
            StructField("credit_card_limit", DoubleType(), True),
            StructField("gender", StringType(), True),
            StructField("id", StringType(), True),
            StructField("registered_on", DateType(), True),
        ]
    )
    
    file_path = Path(base_path) / name
    return spark.read.json(
        str(file_path), schema=profile_schema, dateFormat="yyyyMMdd"
    ).withColumnRenamed("id", "account_id")


def remove_profile_fillna(df: DataFrame) -> DataFrame:
    return df.withColumn("age", F.when(F.col("age") != 118, F.col("age")))


def get_account_age(df: DataFrame, ref_date: str = "2019-01-01") -> DataFrame:
    ref_col = F.to_date(F.lit(ref_date))

    return df.withColumn(
        "account_age", F.datediff(ref_col, F.col("registered_on"))
    ).drop("registered_on")
