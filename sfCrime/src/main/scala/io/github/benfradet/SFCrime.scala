package io.github.benfradet

import java.time.LocalTime
import java.time.format.DateTimeFormatter

import cats.data.Xor
import com.esri.core.geometry._
import io.circe.Decoder
import io.circe.parser.decode
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, VectorAssembler, StringIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.LoggerFactory

import scala.io.Source

object SFCrime {

  private val logger = LoggerFactory.getLogger(getClass)
  private val csvFormat = "com.databricks.spark.csv"

  case class Neighborhood(name: String, polygon: Polygon)
  object Neighborhood {
    implicit val decodeNbhd: Decoder[Neighborhood] = Decoder.instance { c =>
      for {
        name <- c.downField("name").as[String]
        poly <- c.downField("polygon").as[String]
      } yield Neighborhood(name, createPolygonFromWKT(poly))
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.length < 6) {
      System.err.println("Usage: SFCrime <train file> <test file> " +
        "<sunrise/sunset file> <weather file> <neighborhoods file> <output file>")
      System.exit(1)
    }
    val Array(trainFile, testFile, sunsetFile, weatherFile, nbhdsFile, outputFile) = args

    val sc = new SparkContext(new SparkConf().setAppName("SFCrime"))
    val sqlContext = new SQLContext(sc)

    // data loading
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

    // feature engineering
    val enrichFunctions = List(enrichTime, enrichWeekend, enrichAddress,
      enrichDayOrNight(sunsetDF)(_), enrichWeather(weatherDF)(_), enrichNeighborhoods(nbhds)(_))
    val Array(enrichedTrainDF, enrichedTestDF) =
      Array(rawTrainDF, rawTestDF) map (enrichFunctions reduce (_ andThen _))

    // building the pipeline
    val labelColName = "Category"
    val predictedLabelColName = "predictedLabel"
    val featuresColName = "Features"
    val numericFeatColNames = Seq("X", "Y", "temperatureC")
    val categoricalFeatColNames = Seq(
      "DayOfWeek", "PdDistrict", "DayOrNight", "Weekend", "HourOfDay", "Month", "Year",
      "AddressType", "Street", "weather", "Neighborhood"
    )

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

    val indexToString = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol(predictedLabelColName)
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array.concat(
        stringIndexers.toArray,
        Array(labelIndexer, assembler, randomForest, indexToString)
      ))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(labelColName + "Indexed")

    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForest.impurity, Array("entropy", "gini"))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    // training the model
    val cvModel = cv.fit(enrichedTrainDF)

    // making predictions
    val predictions = cvModel
      .transform(enrichedTestDF)
      .select("Id", predictedLabelColName)

    // checking the importance of each feature
    val featureImportances = cvModel
      .bestModel.asInstanceOf[PipelineModel]
      .stages(categoricalFeatColNames.size + 2)
      .asInstanceOf[RandomForestClassificationModel].featureImportances
    assembler.getInputCols
      .zip(featureImportances.toArray)
      .foreach { case (feat, imp) => println(s"feature: $feat, importance: $imp") }

    // retrieving the best model's param
    val bestEstimatorParamMap = cvModel.getEstimatorParamMaps
      .zip(cvModel.avgMetrics)
      .maxBy(_._2)
      ._1
    println(bestEstimatorParamMap)

    // formatting the results according to Kaggle guidelines
    val labels = enrichedTrainDF.select(labelColName).distinct().collect()
      .map { case Row(label: String) => label }
      .sorted

    val labelToVec = (predictedLabel: String) => {
      val array = new Array[Int](labels.length)
      array(labels.indexOf(predictedLabel)) = 1
      array.toSeq
    }

    val schema = StructType(predictions.schema.fields ++ labels.map(StructField(_, IntegerType)))
    val resultDF = sqlContext.createDataFrame(
      predictions.rdd.map { r => Row.fromSeq(
        r.toSeq ++
          labelToVec(r.getAs[String](predictedLabelColName))
      )},
      schema
    )

    // saving the results
    resultDF
      .drop("predictedLabel")
      .coalesce(1)
      .write
      .format(csvFormat)
      .option("header", "true")
      .save(outputFile)
  }

  // add a few time-related features to the original datasets such as:
  //   - the year
  //   - the month
  //   - the hour of day
  val enrichTime = (df: DataFrame) => {
    def dateUDF = udf { (timestamp: String) =>
      val timestampFormatter = DateTimeFormatter.ofPattern("YYYY-MM-dd HH:mm:ss")
      val dateFormat = DateTimeFormatter.ofPattern("YYYY-MM-dd")
      val time = timestampFormatter.parse(timestamp)
      dateFormat.format(time)
    }

    df
      .withColumn("HourOfDay", hour(col("Dates")))
      .withColumn("Month", month(col("Dates")))
      .withColumn("Year", year(col("Dates")))
      .withColumn("TimestampUTC", to_utc_timestamp(col("Dates"), "PST"))
      .withColumn("Date", dateUDF(col("TimestampUTC")))
  }

  // add a weekend feature telling whether or not the crime incident occurred on a weekend
  val enrichWeekend = (df: DataFrame) => {
    def weekendUDF = udf { (dayOfWeek: String) =>
      dayOfWeek match {
        case _ @ ("Friday" | "Saturday" | "Sunday") => 1
        case _ => 0
      }
    }
    df.withColumn("Weekend", weekendUDF(col("DayOfWeek")))
  }

  // add address-related features:
  //   - whether or not the crime incident occurred at an intersection
  //   - a street feature which is the street corresponding to the parsed address
  val enrichAddress = (df: DataFrame) => {
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
    df
      .withColumn("AddressType", addressTypeUDF(col("Address")))
      .withColumn("Street", addressUDF(col("Address")))
  }

  // add a day or night feature based on sunrise and sunset times
  def enrichDayOrNight(sunsetDF: DataFrame)(df: DataFrame): DataFrame = {
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

    df
      .join(sunsetDF, df("Date") === sunsetDF("date"))
      .withColumn("DayOrNight", dayOrNigthUDF(col("TimestampUTC"), col("sunrise"), col("sunset")))
  }

  // add weather-related features:
  //   - the average temperature of the day
  //   - the most occurring weather condition
  def enrichWeather(weatherDF: DataFrame)(df: DataFrame): DataFrame =
    df.join(weatherDF, df("Date") === weatherDF("date"))

  def enrichNeighborhoods(nbhds: Seq[Neighborhood])(df: DataFrame): DataFrame = {
    def nbhdUDF = udf { (lat: Double, lng: Double) =>
      val point = createPointFromWKT(s"POINT($lat $lng)")
      nbhds
        .filter(nbhd => contains(nbhd.polygon, point))
        .map(_.name)
        .headOption match {
          case Some(nbhd) => nbhd
          case None => "SF"
        }
    }
    df.withColumn("Neighborhood", nbhdUDF(col("X"), col("Y")))
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

  def createPolygonFromWKT(wkt: String): Polygon = {
    val geom = OperatorImportFromWkt.local()
      .execute(WktImportFlags.wktImportDefaults, Geometry.Type.Polygon, wkt, null)
    geom.asInstanceOf[Polygon]
  }

  def createPointFromWKT(wkt: String): Point = {
    val geom = OperatorImportFromWkt.local()
      .execute(WktImportFlags.wktImportDefaults, Geometry.Type.Point, wkt, null)
    geom.asInstanceOf[Point]
  }

  def contains(container: Geometry, contained: Geometry): Boolean =
    OperatorContains.local().execute(container, contained, SpatialReference.create(3426), null)
}
