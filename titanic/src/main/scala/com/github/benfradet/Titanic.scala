package com.github.benfradet

import org.apache.log4j.{Logger, Level}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object Titanic {
  val csvFormat = "com.databricks.spark.csv"

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

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

    val trainSchema = StructType(schemaArray)
    val testSchema = StructType(schemaArray.filter(p => p.name != "Survived"))

    // load data
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

    // udf to create a FamilySize column as the sum of the SibSp and Parch columns + 1
    val familySize: ((Int, Int) => Int) = (sibSp: Int, parCh: Int) => sibSp + parCh + 1
    val familySizeUDF = udf(familySize)

    // udf to create a Title column extracting the title from the Name column
    val Pattern = ".*, (.*?)\\..*".r
    val titles = Map(
      "Mrs"    -> "Mrs",
      "Lady"   -> "Mrs",
      "Mme"    -> "Mrs",
      "Ms"     -> "Ms",
      "Miss"   -> "Miss",
      "Mlle"   -> "Miss",
      "Master" -> "Master",
      "Rev"    -> "Rev",
      "Don"    -> "Mr",
      "Sir"    -> "Sir",
      "Dr"     -> "Dr",
      "Col"    -> "Col",
      "Capt"   -> "Col",
      "Major"  -> "Col"
    )
    val title: ((String, String) => String) = {
      case (Pattern(t), sex) => titles.get(t) match {
        case Some(tt) => tt
        case None     =>
          if (sex == "male") "Mr"
          else "Mrs"
      }
      case _ => "Mr"
    }
    val titleUDF = udf(title)

    // udf to create a survived column in the test df
    val survived: (Int => Int) = (_: Int) => 0
    val survivedUDF = udf(survived)

    // TODO: train a model on the age column
    // fill empty values for the age column
    val avgAge = trainDF.select("Age").unionAll(testDF.select("Age"))
      .agg(avg("Age"))
      .collect() match {
        case Array(Row(avg: Double)) => avg
        case _ => 0
      }

    // fill empty values for the fare column
    val avgFare = trainDF.select("Fare").unionAll(testDF.select("Fare"))
      .agg(avg("Fare"))
      .collect() match {
        case Array(Row(avg: Double)) => avg
        case _ => 0
      }


    // map to fill na values
    val fillNAMap = Map(
      "Embarked" -> "S",
      "Fare"     -> avgFare,
      "Age"      -> avgAge
    )

    // process the original DFs
    val trainDFProcessed = trainDF
      .withColumn("FamilySize", familySizeUDF(col("SibSp"), col("Parch")))
      .withColumn("Title", titleUDF(col("Name"), col("Sex")))
      .na.fill(fillNAMap)
      .drop("Name")
      .drop("Ticket")
      .drop("Cabin")

    val testDFProcessed = testDF
      .withColumn("FamilySize", familySizeUDF(col("SibSp"), col("Parch")))
      .withColumn("Title", titleUDF(col("Name"), col("Sex")))
      //.withColumn("Survived", survivedUDF(col("PassengerId")))
      .na.fill(fillNAMap)
      .drop("Name")
      .drop("Ticket")
      .drop("Cabin")

    trainDFProcessed.show(5)
    testDFProcessed.show(5)

    val numericColumnNames = Seq("Age", "SibSp", "Parch", "Fare", "FamilySize")
    val categoricalColumnNames = Seq("Pclass", "Sex", "Embarked", "Title")
    val allColumnNames = numericColumnNames ++ categoricalColumnNames
  }
}
