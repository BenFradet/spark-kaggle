package com.github.benfradet

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types._

object Titanic {
  val csvFormat = "com.databricks.spark.csv"

  def main(args: Array[String]): Unit = {

    if (args.length < 1) {
      System.err.println("Usage: Titanic <filename>")
      System.exit(1)
    }

    val sc = new SparkContext(new SparkConf().setAppName("Titanic"))
    val sqlContext = new SQLContext(sc)

    val filename = args(0)

    val schema = StructType(Array(
      StructField("PassengerId", IntegerType, true),
      StructField("Survived", IntegerType, true),
      StructField("Pclass", IntegerType, true),
      StructField("Name", StringType, true),
      StructField("Sex", StringType, true),
      StructField("Age", FloatType, true),
      StructField("SibSp", IntegerType, true),
      StructField("Parch", IntegerType, true),
      StructField("Ticket", StringType, true),
      StructField("Fare", FloatType, true),
      StructField("Cabin", StringType, true),
      StructField("Embarked", StringType, true)
    ))

    val df = sqlContext.read
      .format(csvFormat)
      .option("header", "true")
      .schema(schema)
      .load(filename)

    val selectedData = df.select("PassengerId", "Survived")
    selectedData.write
      .format(csvFormat)
      .option("header", "true")
      .save("selected.csv")
  }
}
