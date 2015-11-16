package com.github.benfradet

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

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
      .drop("PassengerId")
      .drop("Name")
      .drop("Ticket")
      .drop("Cabin")

    val testDF = sqlContext.read
      .format(csvFormat)
      .option("header", "true")
      .schema(testSchema)
      .load(args(1))
      .drop("PassengerId")

    // create a column the sum as the SibSp and Parch columns
    val familySize: ((Int, Int) => Int) = (sibsp: Int, parch: Int) => sibsp + parch + 1
    val sumUDF = udf(familySize)

    val trainDFWithFamilySize = trainDF
      .withColumn("FamilySize", sumUDF(col("SibSp"), col("Parch")))
    val testDFWithFamilySize = testDF
      .withColumn("FamilySize", sumUDF(col("SibSp"), col("Parch")))

    // TODO: train a model on the age column
    // fill empty values
    val avgAge = trainDF.select("Age").unionAll(testDF.select("Age"))
      .agg(avg("Age"))
      .collect() match {
        case Array(Row(avg: Double)) => avg
        case _ => 0
      }

    val avgFare = trainDF.select("Fare").unionAll(testDF.select("Fare"))
      .agg(avg("Fare"))
      .collect() match {
        case Array(Row(avg: Double)) => avg
        case _ => 0
      }

    val fillNAMap = Map(
      "Embarked" -> "S",
      "Fare"     -> avgFare,
      "Age"      -> avgAge
    )

    val trainDFFilled = trainDFWithFamilySize.na.fill(fillNAMap)
    val testDFFilled = testDFWithFamilySize.na.fill(fillNAMap)

    val numericColumnNames = Seq("Age", "SibSp", "Parch", "Fare")
    val categoricalColumnNames = Seq("Pclass", "Sex", "Embarked")

    val selectedData = trainDFWithFamilySize.select("SibSp", "Parch", "FamilySize")
    selectedData.write
      .format(csvFormat)
      .option("header", "true")
      .save(args(2))
  }
}
