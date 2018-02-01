import Dependencies._

val SparkVersion = "2.2.0"

organization := "com.github.takemikami"
name         := "selica"
version      := "0.0.2"

scalaVersion := "2.11.11"

licenses := Seq("APL2" -> url("http://www.apache.org/licenses/LICENSE-2.0.txt"))
homepage := Some(url("https://github.com/takemikami/selica"))

libraryDependencies ++= Seq (
  "com.atilika.kuromoji" % "kuromoji-ipadic" % "0.9.0",
  scalaTest % Test,
  "org.apache.spark" %% "spark-mllib" % SparkVersion % "provided"
)

// test
coverageMinimum := 75
coverageFailOnMinimum := true

// publish information
publishMavenStyle := true
publishArtifact in Test := false
publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)

scmInfo := Some(
  ScmInfo(
    url("https://github.com/takemikami/selica"),
    "scm:git:git@github.com:takemikami/selica.git"
  )
)

developers := List(
  Developer(
    id    = "takemikami",
    name  = "Takeshi Mikami",
    email = "takeshi.mikami@gmail.com",
    url   = url("https://github.com/takemikami")
  )
)
