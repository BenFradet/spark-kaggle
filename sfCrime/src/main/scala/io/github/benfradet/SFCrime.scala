package io.github.benfradet

import org.apache.log4j.{Logger, Level}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, VectorAssembler, StringIndexer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

object SFCrime {

  val csvFormat = "com.databricks.spark.csv"

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)

    if (args.length < 3) {
      System.err.println("Usage: SFCrime <train file> <test file> <output file>")
      System.exit(1)
    }

    val sc = new SparkContext(new SparkConf().setAppName("Titanic"))
    val sqlContext = new SQLContext(sc)

    val (rawTrainDF, rawTestDF) = loadData(args(0), args(1), sqlContext)

    // extra features:
    // - day/night
    // - weather:
    //   min date: 2003-01-01 00:01:00.0, max date: 2015-05-13 23:53:00.0

    val (enrichedTrainDF, enrichedTestDF) = enrichData(rawTrainDF, rawTestDF)

    val labelColName = "Category"
    val predictedLabelColName = "predictedLabel"
    val featuresColName = "Features"
    val numericFeatColNames = Seq("X", "Y")
    val categoricalFeatColNames = Seq("DayOfWeek", "PdDistrict",
      "Weekend", "HourOfDay", "Month", "Year", "AddressType", "AddressParsed")

    val allData = enrichedTrainDF
      .select((numericFeatColNames ++ categoricalFeatColNames).map(col): _*)
      .unionAll(enrichedTestDF
        .select((numericFeatColNames ++ categoricalFeatColNames).map(col): _*))
    allData.cache()

    val stringIndexers = categoricalFeatColNames.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(colName + "Indexed")
        .fit(allData)
    }

    val labelIndexer = new StringIndexer()
      .setInputCol(labelColName)
      .setOutputCol(labelColName + "Indexed")
      .fit(enrichedTrainDF)

    val assembler = new VectorAssembler()
      .setInputCols((categoricalFeatColNames.map(_ + "Indexed") ++ numericFeatColNames).toArray)
      .setOutputCol(featuresColName)

    val randomForest = new RandomForestClassifier()
      .setLabelCol(labelColName + "Indexed")
      .setFeaturesCol(featuresColName)
      .setMaxDepth(10)
      .setMaxBins(2089)
      .setImpurity("entropy")

    val indexToString = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol(predictedLabelColName)
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array.concat(
        stringIndexers.toArray,
        Array(labelIndexer, assembler, randomForest, indexToString)
      ))

    val pipelineModel = pipeline.fit(enrichedTrainDF)

    val featureImportances =
      pipelineModel.stages(10).asInstanceOf[RandomForestClassificationModel].featureImportances
    assembler.getInputCols
      .zip(featureImportances.toArray)
      .foreach { case (feat, imp) => println(s"feature: $feat, importance: $imp") }

    val labels = enrichedTrainDF.select(labelColName).distinct().collect()
      .map { case Row(label: String) => label }
      .sorted

    val labelToVec = (predictedLabel: String) => {
      val array = new Array[Int](labels.length)
      array(labels.indexOf(predictedLabel)) = 1
      array.toSeq
    }

    val predictions = pipelineModel
      .transform(enrichedTestDF)
      .select("Id", predictedLabelColName)

    val schema = StructType(predictions.schema.fields ++ labels.map(StructField(_, IntegerType)))
    val predictionsRDD = predictions.rdd.map { r => Row.fromSeq(
      r.toSeq ++
      labelToVec(r.getAs[String](predictedLabelColName))
    )}

    sqlContext.createDataFrame(predictionsRDD, schema)
      .drop("predictedLabel")
      .coalesce(1)
      .write
      .format(csvFormat)
      .option("header", "true")
      .save(args(2))
  }

  def enrichData(trainDF: DataFrame, predictDF: DataFrame): (DataFrame, DataFrame) = {
    def addressTypeUDF = udf { (address: String) =>
      if (address contains "/") "Intersection"
      else "Street"
    }

    val streetRegex = """\d{1,4} Block of (.+)""".r
    val intersectionRegex = """(.+) / (.+)""".r
    def addressUDF = udf { (address: String) =>
      streetRegex findFirstIn address match {
        case Some(streetRegex(s)) => s
        case None => intersectionRegex findFirstIn address match {
          case Some(intersectionRegex(s1, s2)) => if (s1 < s2) s1 else s2
          case None => address
        }
      }
    }

    def weekendUDF = udf { (dayOfWeek: String) =>
      dayOfWeek match {
        case _ @ ("Friday" | "Saturday" | "Sunday") => 1
        case _ => 0
      }
    }

    (
      trainDF
        .withColumn("AddressType", addressTypeUDF(col("Address")))
        .withColumn("AddressParsed", addressUDF(col("Address")))
        .withColumn("Weekend", weekendUDF(col("DayOfWeek")))
        .withColumn("HourOfDay", hour(col("Dates")))
        .withColumn("Month", month(col("Dates")))
        .withColumn("Year", year(col("Dates"))),
      predictDF
        .withColumn("AddressType", addressTypeUDF(col("Address")))
        .withColumn("AddressParsed", addressUDF(col("Address")))
        .withColumn("Weekend", weekendUDF(col("DayOfWeek")))
        .withColumn("HourOfDay", hour(col("Dates")))
        .withColumn("Month", month(col("Dates")))
        .withColumn("Year", year(col("Dates")))
      )
  }

  def loadData(
    trainFile: String,
    testFile: String,
    sqlContext: SQLContext
  ): (DataFrame, DataFrame) = {
    val schemaArray = Array(
      StructField("Id", LongType),
      StructField("Dates", TimestampType),
      StructField("Category", StringType), // target variable
      StructField("Descript", StringType),
      StructField("DayOfWeek", StringType),
      StructField("PdDistrict", StringType),
      StructField("Resolution", StringType),
      StructField("Address", StringType),
      StructField("X", DoubleType),
      StructField("Y", DoubleType)
    )

    val trainSchema = StructType(schemaArray.filterNot(_.name == "Id"))
    val testSchema = StructType(schemaArray.filterNot { p =>
      Seq("Category", "Descript", "Resolution") contains p.name
    })

    val trainDF = sqlContext.read
      .format(csvFormat)
      .option("header", "true")
      .schema(trainSchema)
      .load(trainFile)

    val testDF = sqlContext.read
      .format(csvFormat)
      .option("header", "true")
      .schema(testSchema)
      .load(testFile)

    (trainDF, testDF)
  }
}
