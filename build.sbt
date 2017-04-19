lazy val sparkVersion = "2.1.0"
lazy val circeVersion = "0.6.1"
lazy val slf4jVersion = "1.7.21"

lazy val buildSettings = Seq(
  organization := "io.github.benfradet",
  version := "1.0",
  scalaVersion := "2.11.8",
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core",
    "org.apache.spark" %% "spark-mllib",
    "org.apache.spark" %% "spark-sql"
  ).map(_ % sparkVersion % "provided"),
  scalacOptions ++= compilerOptions
)

lazy val compilerOptions = Seq(
  "-deprecation",
  "-encoding", "UTF-8",
  "-feature",
  "-language:existentials",
  "-language:higherKinds",
  "-language:implicitConversions",
  "-unchecked",
  "-Yno-adapted-args",
  "-Ywarn-dead-code",
  "-Ywarn-numeric-widen",
  "-Xfuture",
  "-Xlint"
)

lazy val kaggle = project.in(file("."))
  .settings(moduleName := "spark-kaggle")
  .settings(buildSettings)
  .aggregate(titanic, sfCrime)

lazy val titanic = project
  .settings(moduleName := "titanic")
  .settings(buildSettings)

lazy val sfCrime = project
  .settings(moduleName := "sfCrime")
  .settings(buildSettings)
  .settings(libraryDependencies ++= Seq(
    "io.circe" %% "circe-core",
    "io.circe" %% "circe-generic",
    "io.circe" %% "circe-parser"
  ).map(_ % circeVersion) ++ Seq(
    "com.twitter" %% "util-core" % "6.34.0",
    "com.esri.geometry" % "esri-geometry-api" % "1.2.1"
  ) ++ Seq(
    "org.slf4j" % "slf4j-api",
    "org.slf4j" % "slf4j-log4j12"
  ).map(_ % slf4jVersion))
