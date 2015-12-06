package com.github.benfradet

import org.apache.log4j.{Logger, Level}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, VectorIndexer, VectorAssembler, StringIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object Titanic {

  val csvFormat = "com.databricks.spark.csv"

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)

    if (args.length < 3) {
      System.err.println("Usage: Titanic <train file> <predict file> <output file>")
      System.exit(1)
    }

    val sc = new SparkContext(new SparkConf().setAppName("Titanic"))
    val sqlContext = new SQLContext(sc)

    val (dataDFRaw, predictDFRaw) = loadData(args(0), args(1), sqlContext)

    val (dataDFExtra, predictDFExtra) = createExtraFeatures(dataDFRaw, predictDFRaw)

    val (dataDFCompleted, predictDFCompleted) = fillNAValues(dataDFExtra, predictDFExtra)

    val numericFeatColNames = Seq("Age", "SibSp", "Parch", "Fare", "FamilySize")
    val categoricalFeatColNames = Seq("Pclass", "Sex", "Embarked", "Title")
    val idxdCategoricalFeatColName = categoricalFeatColNames.map(_ + "Indexed")
    val allFeatColNames = numericFeatColNames ++ categoricalFeatColNames
    val allIdxdFeatColNames = numericFeatColNames ++ idxdCategoricalFeatColName

    val labelColName = "SurvivedString"
    val featColName = "Features"
    val idColName = "PassengerId"

    val allPredictColNames = allFeatColNames ++ Seq(idColName)

    val dataDFFiltered = dataDFCompleted.select(labelColName, allPredictColNames: _*)
    val predictDFFiltered = predictDFCompleted.select(labelColName, allPredictColNames: _*)

    val (dataDFIndexed, predictDFIndexed) =
      indexCategoricalFeatures(dataDFFiltered, predictDFFiltered, categoricalFeatColNames)

    // vector assembler
    val assembler = new VectorAssembler()
      .setInputCols(Array(allIdxdFeatColNames: _*))
      .setOutputCol(featColName)

    // transform the dataframe with the previously defined assembler
    val dataDFAssembled = assembler
      .transform(dataDFIndexed)
    dataDFAssembled.cache()
    val predictDFAssembled = assembler
      .transform(predictDFIndexed)
    predictDFAssembled.cache()

    val allData = dataDFAssembled.unionAll(predictDFAssembled)
    allData.cache()

    val idxdLabelColName = "SurvivedIndexed"

    // index classes
    val labelIndexer = new StringIndexer()
      .setInputCol(labelColName)
      .setOutputCol(idxdLabelColName)
      .fit(allData)

    val randomForest = new RandomForestClassifier()
      .setLabelCol(idxdLabelColName)
      .setFeaturesCol(featColName)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // define the order of the operations to be performed
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, randomForest, labelConverter))

    // grid of values to perform cross validation on
    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForest.maxBins, Array(25, 28, 31))
      .addGrid(randomForest.maxDepth, Array(4, 6, 8))
      .addGrid(randomForest.impurity, Array("entropy", "gini"))
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol(idxdLabelColName)

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    // train the model
    val crossValidatorModel = cv.fit(dataDFAssembled)

    // make predictions
    val predictions = crossValidatorModel.transform(predictDFAssembled)

    predictions.select("predictedLabel", "rawPrediction", featColName).show(5, false)

    predictions
      .withColumn("Survived", col("predictedLabel"))
      .select("PassengerId", "Survived")
      .write
      .format(csvFormat)
      .option("header", "true")
      .save(args(2))
  }

  def indexCategoricalFeatures(
    trainDF: DataFrame,
    testDF: DataFrame,
    categoricalFeatColNames: Seq[String]
  ): (DataFrame, DataFrame) = {
    val dataFiltered = trainDF.unionAll(testDF)

    var oldTrainDF = trainDF
    var newTrainDF = trainDF
    var oldTestDF = testDF
    var newTestDF = testDF
    categoricalFeatColNames
      .map(colName => new StringIndexer()
      .setInputCol(colName)
      .setOutputCol(colName + "Indexed")
      .fit(dataFiltered)
      )
      .foreach { stringIndexer =>
      newTrainDF = stringIndexer.transform(oldTrainDF)
      newTestDF = stringIndexer.transform(oldTestDF)
      oldTrainDF = newTrainDF
      oldTestDF = newTestDF
    }

    (newTrainDF, newTestDF)
  }

  def fillNAValues(trainDF: DataFrame, testDF: DataFrame): (DataFrame, DataFrame) = {
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
      "Fare"     -> avgFare,
      "Age"      -> avgAge
    )

    // udf to fill empty embarked string with S corresponding to Southampton
    val embarked: (String => String) = {
      case "" => "S"
      case a  => a
    }
    val embarkedUDF = udf(embarked)

    val newTrainDF = trainDF
      .na.fill(fillNAMap)
      .withColumn("Embarked", embarkedUDF(col("Embarked")))

    val newTestDF = testDF
      .na.fill(fillNAMap)
      .withColumn("Embarked", embarkedUDF(col("Embarked")))

    (newTrainDF, newTestDF)
  }

  def createExtraFeatures(trainDF: DataFrame, testDF: DataFrame): (DataFrame, DataFrame) = {
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

    val newTrainDF = trainDF
      .withColumn("FamilySize", familySizeUDF(col("SibSp"), col("Parch")))
      .withColumn("Title", titleUDF(col("Name"), col("Sex")))
      .withColumn("SurvivedString", trainDF("Survived").cast(StringType))
    val newTestDF = testDF
      .withColumn("FamilySize", familySizeUDF(col("SibSp"), col("Parch")))
      .withColumn("Title", titleUDF(col("Name"), col("Sex")))
      .withColumn("SurvivedString", lit("0").cast(StringType))

    (newTrainDF, newTestDF)
  }

  def loadData(
    trainFile: String,
    testFile: String,
    sqlContext: SQLContext
  ): (DataFrame, DataFrame) = {
    val nullable = true
    val schemaArray = Array(
      StructField("PassengerId", IntegerType, nullable),
      StructField("Survived", IntegerType, nullable),
      StructField("Pclass", IntegerType, nullable),
      StructField("Name", StringType, nullable),
      StructField("Sex", StringType, nullable),
      StructField("Age", FloatType, nullable),
      StructField("SibSp", IntegerType, nullable),
      StructField("Parch", IntegerType, nullable),
      StructField("Ticket", StringType, nullable),
      StructField("Fare", FloatType, nullable),
      StructField("Cabin", StringType, nullable),
      StructField("Embarked", StringType, nullable)
    )

    val trainSchema = StructType(schemaArray)
    val testSchema = StructType(schemaArray.filter(p => p.name != "Survived"))

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
