package com.github.takemikami.selica.ml.recommendation

import org.apache.spark.sql.SparkSession
import org.scalatest._
import Matchers._
import com.github.takemikami.selica._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, MinMaxScaler, StandardScaler, StringIndexer}

class ItemBasedCollaborativeFilteringSpec extends FlatSpec with BeforeAndAfter {

  private var sparkSession: SparkSession = _
  before {
    sparkSession = SparkSessionForUnitTest.getSession()
  }
  after {
  }

  "ItemBasedCollaborativeFiltering" should "can fit & transform" in {
    val spark = sparkSession
    import spark.implicits._

    val ratings = Seq(
      ("u0", "i0", 1.0, 1L),
      ("u0", "i1", 0.1, 1L),
      ("u0", "i2", 0.1, 1L),
      ("u1", "i0", 1.0, 1L),
      ("u1", "i1", 0.5, 1L),
      ("u2", "i1", 0.5, 1L),
      ("u3", "i2", 0.5, 1L)
    ).toDF("userId", "itemId", "rating", "timestamp")

    // fitting
    val cf = new com.github.takemikami.selica.ml.recommendation.ItemBasedCollaborativeFiltering()
        .setUserCol("userId")
        .setItemCol("itemId")
        .setRatingCol("rating")
    val model = cf.fit(ratings)

    model.itemSimilarity.numRows() shouldEqual 3

    // transform
    val df = model
      .setPredictionCol("pred")
      .transform(ratings)
    df.collect()

    df.columns.contains("userId") shouldEqual true
    df.columns.contains("itemId") shouldEqual true
    df.columns.contains("pred") shouldEqual true
  }


  "ItemBasedCollaborativeFiltering" should "can fit under pipeline" in {
    val spark = sparkSession
    import spark.implicits._

    val ratings = Seq(
      ("u0", "i0", 1.0, 1L),
      ("u0", "i1", 0.1, 1L),
      ("u0", "i2", 0.1, 1L),
      ("u1", "i0", 1.0, 1L),
      ("u1", "i1", 0.5, 1L),
      ("u2", "i1", 0.5, 1L),
      ("u3", "i2", 0.5, 1L)
    ).toDF("userId", "itemId", "rating", "timestamp")

    // fitting with pipeline
    val cf = new com.github.takemikami.selica.ml.recommendation.ItemBasedCollaborativeFiltering()
      .setUserCol("userId")
      .setItemCol("itemId")
      .setRatingCol("rating")

    val pipeline = new Pipeline()
      .setStages(Array(cf))

    val model = pipeline.fit(ratings)

    // transform
    val df = model.transform(ratings)
    df.collect()

    df.columns.contains("userId") shouldEqual true
    df.columns.contains("itemId") shouldEqual true
    df.columns.contains("prediction") shouldEqual true
  }

}
