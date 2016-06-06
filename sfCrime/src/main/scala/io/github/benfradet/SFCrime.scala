package io.github.benfradet

import java.time.LocalTime
import java.time.format.DateTimeFormatter

import cats.data.Xor
import com.esri.core.geometry.ogc.OGCGeometry
import io.circe.generic.auto._
import io.circe.parser.decode
import org.apache.log4j.{Logger, Level}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, VectorAssembler, StringIndexer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.LoggerFactory

import scala.io.Source

object SFCrime {

  private val logger = LoggerFactory.getLogger(getClass)
  private val csvFormat = "com.databricks.spark.csv"
  case class Neighborhood(name: String, polygon: String)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)

    if (args.length < 6) {
      System.err.println("Usage: SFCrime <train file> <test file> " +
        "<sunrise/sunset file> <weather file> <neighborhoods file> <output file>")
      System.exit(1)
    }
    val Array(trainFile, testFile, sunsetFile, weatherFile, nbhdsFile, outputFile) = args

    val sc = new SparkContext(new SparkConf().setAppName("Titanic"))
    val sqlContext = new SQLContext(sc)

    val (rawTrainDF, rawTestDF) = loadData(trainFile, testFile, sqlContext)
    val sunsetDF = {
      val rdd = sc.wholeTextFiles(sunsetFile).map(_._2)
      sqlContext.read.json(rdd)
    }
    val weatherDF = {
      val rdd = sc.wholeTextFiles(weatherFile).map(_._2)
      sqlContext.read.json(rdd)
    }
    val nbhds = decode[Seq[Neighborhood]](Source.fromFile(nbhdsFile).getLines().mkString) match {
      case Xor.Right(l) => l
      case Xor.Left(e) =>
        logger.error("Couldn't parse the neighborhoods file", e)
        Seq.empty[Neighborhood]
    }

    val (enrichedTrainDF, enrichedTestDF) =
      enrichData(rawTrainDF, rawTestDF, sunsetDF, weatherDF, nbhds)

    val labelColName = "Category"
    val predictedLabelColName = "predictedLabel"
    val featuresColName = "Features"
    val numericFeatColNames = Seq("X", "Y", "temperatureC")
    val categoricalFeatColNames = Seq("DayOfWeek", "PdDistrict", "DayOrNight",
      "Weekend", "HourOfDay", "Month", "Year",
      "AddressType", "AddressParsed",
      "weather", "Neighborhood")

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

    val featureImportances = pipelineModel.stages(categoricalFeatColNames.size + 2)
      .asInstanceOf[RandomForestClassificationModel].featureImportances
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
      .save(outputFile)
  }

  def enrichData(
    trainDF: DataFrame,
    predictDF: DataFrame,
    sunsetDF: DataFrame,
    weatherDF: DataFrame,
    nbhds: Seq[Neighborhood]
  ): (DataFrame, DataFrame) = {
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

    def dateUDF = udf { (timestamp: String) =>
      val timestampFormatter = DateTimeFormatter.ofPattern("YYYY-MM-dd HH:mm:ss")
      val dateFormat = DateTimeFormatter.ofPattern("YYYY-MM-dd")
      val time = timestampFormatter.parse(timestamp)
      dateFormat.format(time)
    }

    def dayOrNigthUDF = udf { (timestampUTC: String, sunrise: String, sunset: String) =>
      val timestampFormatter = DateTimeFormatter.ofPattern("YYYY-MM-dd HH:mm:ss")
      val timeFormatter = DateTimeFormatter.ofPattern("h:mm:ss a")
      val time = LocalTime.parse(timestampUTC, timestampFormatter)
      val sunriseTime = LocalTime.parse(sunrise, timeFormatter)
      val sunsetTime = LocalTime.parse(sunset, timeFormatter)
      if (sunriseTime.compareTo(sunsetTime) > 0) {
        if (time.compareTo(sunsetTime) > 0 && time.compareTo(sunriseTime) < 0) {
          "Night"
        } else {
          "Day"
        }
      } else {
        if (time.compareTo(sunriseTime) > 0 && time.compareTo(sunsetTime) < 0) {
          "Day"
        } else {
          "Night"
        }
      }
    }

    def nbhdUDF = udf { (lat: Double, lng: Double) =>
      val point = OGCGeometry.fromText(s"POINT($lat $lng)")
      logger.info(point.asText())
      nbhds
        .map(n => (n.name, OGCGeometry.fromText(n.polygon)))
        .filter { case (name, polygon) => polygon.contains(point) }
        .map(_._1)
        .headOption match {
          case Some(nbhd) => nbhd
          case None => "SF"
        }
    }

    val Array(newTrainDF, newPredictDF) = Array(trainDF, predictDF).map { df =>
      val dfWithDate = df
        .withColumn("TimestampUTC", to_utc_timestamp(col("Dates"), "PST"))
        .withColumn("Date", dateUDF(col("TimestampUTC")))

      val dfJoined = dfWithDate
        .join(sunsetDF, dfWithDate("Date") === sunsetDF("date"))
        .withColumn("DayOrNight", dayOrNigthUDF(col("TimestampUTC"), col("sunrise"), col("sunset")))
        .join(weatherDF, dfWithDate("Date") === weatherDF("date"))

      dfJoined
        .withColumn("AddressType", addressTypeUDF(col("Address")))
        .withColumn("AddressParsed", addressUDF(col("Address")))
        .withColumn("Weekend", weekendUDF(col("DayOfWeek")))
        .withColumn("HourOfDay", hour(col("Dates")))
        .withColumn("Month", month(col("Dates")))
        .withColumn("Year", year(col("Dates")))
        .withColumn("Neighborhood", nbhdUDF(col("X"), col("Y")))
    }

    (newTrainDF, newPredictDF)
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
