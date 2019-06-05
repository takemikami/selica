import Dependencies._

val SparkVersion = "2.4.3"

organization := "com.github.takemikami"
name         := "selica"
version      := "0.0.3-SNAPSHOT"

scalaVersion := "2.12.8"

licenses := Seq("APL2" -> url("http://www.apache.org/licenses/LICENSE-2.0.txt"))
homepage := Some(url("https://github.com/takemikami/selica"))

lazy val core = (project in file("core"))
  .settings(
    libraryDependencies ++= Seq (
    "com.atilika.kuromoji" % "kuromoji-ipadic" % "0.9.0",
    scalaTest % Test,
    "org.apache.spark" %% "spark-mllib" % SparkVersion % "provided"
  )
)

// test
scapegoatVersion in ThisBuild := "1.3.8"
scalaBinaryVersion in ThisBuild := "2.12"
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
