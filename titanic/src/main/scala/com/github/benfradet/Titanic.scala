package com.github.benfradet

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types._

object Titanic {
  val csvFormat = "com.databricks.spark.csv"

  def main(args: Array[String]): Unit = {

    if (args.length < 3) {
      System.err.println("Usage: Titanic <train file> <test file> <output file>")
      System.exit(1)
    }

    val sc = new SparkContext(new SparkConf().setAppName("Titanic"))
    val sqlContext = new SQLContext(sc)

    val schemaArray = Array(
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
    )

    val testSchema = StructType(schemaArray.filter(p => p.name != "Survived"))

    val trainSchema = StructType(schemaArray)

    val trainDF = sqlContext.read
      .format(csvFormat)
      .option("header", "true")
      .schema(trainSchema)
      .load(args(0))

    val testDF = sqlContext.read
      .format(csvFormat)
      .option("header", "true")
      .schema(testSchema)
      .load(args(1))

    val selectedData = trainDF.select("PassengerId", "Survived")
    selectedData.write
      .format(csvFormat)
      .option("header", "true")
      .save(args(2))
  }
}
