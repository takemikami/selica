package com.github.takemikami.selica

import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}

object SparkSessionForUnitTest {
  val level = Level.WARN
  Logger.getLogger("org").setLevel(level)
  Logger.getLogger("akka").setLevel(level)

  private val master = "local[2]"
  private val appName = "SparkCF"

  private var sparkSession: SparkSession = _

  def getSession(): SparkSession = {
    sparkSession = SparkSession
      .builder()
      .master(master)
      .appName(appName)
      .getOrCreate()
    sparkSession
  }

}
