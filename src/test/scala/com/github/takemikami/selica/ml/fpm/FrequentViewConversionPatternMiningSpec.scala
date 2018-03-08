package com.github.takemikami.selica.ml.fpm

import org.scalatest._
import Matchers._
import com.github.takemikami.selica.SparkSessionForUnitTest
import org.apache.spark.sql.SparkSession

class FrequentViewConversionPatternMiningSpec extends FlatSpec with BeforeAndAfter {

  private var sparkSession: SparkSession = _
  before {
    sparkSession = SparkSessionForUnitTest.getSession()
  }
  after {
  }

  "FrequentViewConversionPatternMining" should "can fit & transform" in {
    val spark = sparkSession
    import spark.implicits._

    val dataset = spark.createDataset(Seq(
      (1, Array(1,2,3), Array(1)),
      (1, Array(4,5,6), Array(4,6)),
      (2, Array(2,3,4), Array(1))
    )).toDF("userId", "antecedent", "consequent")

    val fpm = new com.github.takemikami.selica.ml.fpm.FrequentViewConversionPatternMining()
      .setAntecedentCol("antecedent")
      .setConsequentCol("consequent")
    val model = fpm.fit(dataset)

    model.ruleDataFrame.columns.contains("item_i") shouldEqual true
    model.ruleDataFrame.columns.contains("item_j") shouldEqual true
    model.ruleDataFrame.columns.contains("support") shouldEqual true
    model.ruleDataFrame.columns.contains("confidence") shouldEqual true
    model.ruleDataFrame.columns.contains("lift") shouldEqual true

    // support & confidence <= 1.0
    model.ruleDataFrame.select("support").rdd.map(r => r(0)).collect().foreach { x =>
      x match {
        case (x: Double) => x should be <= 1.0
        case _ => fail
      }
    }
    model.ruleDataFrame.select("confidence").rdd.map(r => r(0)).collect().foreach { x =>
      x match {
        case (x: Double) => x should be <= 1.0
        case _ => fail
      }
    }

    println(model.ruleDataFrame.select("item_i", "item_j", "support", "confidence", "lift").show())

    val predict = model.transform(dataset)

    println(predict.show())

    predict.columns.contains("prediction") shouldEqual true
    predict.count() shouldEqual 3

  }

  "FrequentViewConversionPatternMining" should "can fit with minimal support confidence" in {
    val spark = sparkSession
    import spark.implicits._

    val dataset = spark.createDataset(Seq(
      (1, Array(1,2,3), Array(1)),
      (1, Array(4,5,6), Array(4,6)),
      (2, Array(2,3,4), Array(1))
    )).toDF("userId", "antecedent", "consequent")

    val fpm = new com.github.takemikami.selica.ml.fpm.FrequentViewConversionPatternMining()
      .setAntecedentCol("antecedent")
      .setConsequentCol("consequent")
      .setMinSupport(0.5)
      .setMinConfidence(0.6)

    val model = fpm.fit(dataset)

    // support & confidence <= 1.0
    model.ruleDataFrame.select("support").rdd.map(r => r(0)).collect().foreach { x =>
      x match {
        case (x: Double) => {
          x should be <= 1.0
          x should be > 0.5
        }
        case _ => fail
      }
    }
    model.ruleDataFrame.select("confidence").rdd.map(r => r(0)).collect().foreach { x =>
      x match {
        case (x: Double) => {
          x should be <= 1.0
          x should be > 0.6
        }
        case _ => fail
      }
    }

    println(model.ruleDataFrame.select("item_i", "item_j", "support", "confidence", "lift").show())
  }

  "FrequentViewConversionPatternMining" should "have required column" in {
    val spark = sparkSession
    import spark.implicits._

    val dataset = spark.createDataset(Seq(
      (1, Array(1,2,3), Array(1)),
      (1, Array(4,5,6), Array(4,6)),
      (2, Array(2,3,4), Array(1))
    ))
    val fpm = new com.github.takemikami.selica.ml.fpm.FrequentViewConversionPatternMining()
      .setAntecedentCol("antecedent")
      .setConsequentCol("consequent")

    // fitting columns check
    an[IllegalArgumentException] should be thrownBy fpm.transformSchema(dataset.toDF("userId", "non_antecedent", "consequent").schema)
    an[IllegalArgumentException] should be thrownBy fpm.transformSchema(dataset.toDF("userId", "antecedent", "non_consequent").schema)
    val schemaFitted = fpm.transformSchema(dataset.toDF("userId", "antecedent", "consequent").schema)
    val model = fpm.fit(dataset.toDF("userId", "antecedent", "consequent"))

    // transform columns check
    an[IllegalArgumentException] should be thrownBy model.transformSchema(dataset.toDF("userId", "non_antecedent", "consequent").schema)
    an[IllegalArgumentException] should be thrownBy model.transformSchema(dataset.toDF("userId", "antecedent", "prediction").schema)
    val schemaTransformed = an[IllegalArgumentException] should be thrownBy model.transformSchema(dataset.toDF("userId", "antecedent", "prediction").schema)
  }

}
