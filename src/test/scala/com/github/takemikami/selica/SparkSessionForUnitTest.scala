package com.github.takemikami.selica

import org.apache.spark.sql.SparkSession

object SparkSessionForUnitTest {
  private val master = "local[2]"
  private val appName = "SparkCF"

  private var sparkSession: SparkSession = _

  def getSession(): SparkSession = {
    sparkSession = SparkSession
      .builder()
      .master(master)
      .appName(appName)
      .getOrCreate()
    return sparkSession
  }

}
