import Dependencies._

val Organization = "com.github.takemikami"
val Name = "selica"
val SelicaVersion = "0.0.1-SNAPSHOT"
val SparkVersion = "2.2.0"

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := Organization,
      scalaVersion := "2.11.11",
      version      := SelicaVersion
    )),
    name := Name,
    libraryDependencies ++= Seq (
      scalaTest % Test,
      "org.apache.spark" %% "spark-mllib" % SparkVersion % "provided"
    )
  )
