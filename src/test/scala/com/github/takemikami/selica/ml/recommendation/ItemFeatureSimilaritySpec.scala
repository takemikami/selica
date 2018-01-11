package com.github.takemikami.selica.ml.recommendation


import com.github.takemikami.selica.SparkSessionForUnitTest
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import org.apache.spark.ml.linalg.{SparseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructField, StructType}
import Matchers._

class ItemFeatureSimilaritySpec extends FlatSpec with BeforeAndAfter {

  private var sparkSession: SparkSession = _
  before {
    sparkSession = SparkSessionForUnitTest.getSession()
  }
  after {
  }

  "ItemFeatureSimilarity" should "can fitting by features" in {
    val spark = sparkSession
    import spark.implicits._

    val ratings = Seq(
      ("i0", Vectors.sparse(3, Array(1, 2), Array(0.1, 0.2))),
      ("i1", Vectors.sparse(3, Array(0, 2), Array(0.1, 0.2))),
      ("i2", Vectors.sparse(3, Array(1, 2), Array(0.2, 0.2))),
      ("i3", Vectors.sparse(3, Array(1, 2), Array(0.2, 0.2))),
      ("i4", Vectors.sparse(3, Array(1, 2), Array(0.2, 0.2)))
    ).toDF("itemId", "features")

    // fitting
    val ifs = new com.github.takemikami.selica.ml.recommendation.ItemFeatureSimilarity()
      .setItemCol("itemId")
      .setFeaturesCol("features")
    val model = ifs.fit(ratings)

    // similarity matrix size
    model.itemSimilarity.numRows() shouldEqual 5

    // similarity dataframe schema
    val cols = model.similarityDataFrame.schema.map { f => f.name }.toList
    cols should contain ("item_i")
    cols should contain ("item_j")
    cols should contain ("similarity")

    // similarity dataframe value
    model.similarityDataFrame.select("similarity").rdd.map { row =>
      row(0) match { case x: Double => (x <= 1) shouldEqual true }
    }

  }

  "ItemFeatureSimilarity" should "can fitting by densevector features" in {
    val spark = sparkSession
    import spark.implicits._

    val ratings = Seq(
      ("hokkaido", Vectors.dense(Array(5320523, 78420.77))),
      ("tokyo", Vectors.dense(Array(13742906, 2191.0))),
      ("kyoto", Vectors.dense(Array(2599313, 4612.19))),
      ("osaka", Vectors.dense(Array(8831642, 1905.14))),
      ("okinawa", Vectors.dense(Array(1443802, 2281.14)))
    ).toDF("itemId", "features")

    // fitting
    val ifs = new com.github.takemikami.selica.ml.recommendation.ItemFeatureSimilarity()
      .setItemCol("itemId")
      .setFeaturesCol("features")
    val model = ifs.fit(ratings)

  }

  "ItemFeatureSimilarity" should "can fitting by hashed features" in {
    val spark = sparkSession
    import spark.implicits._

    val ratings = Seq(
      ("hokkaido", Vectors.dense(Array(5320523, 78420.77))),
      ("tokyo", Vectors.dense(Array(13742906, 2191.0))),
      ("kyoto", Vectors.dense(Array(2599313, 4612.19))),
      ("osaka", Vectors.dense(Array(8831642, 1905.14))),
      ("okinawa", Vectors.dense(Array(1443802, 2281.14)))
    ).toDF("itemId", "features")

    // feature hashing
    import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
    val brp = new BucketedRandomProjectionLSH().setBucketLength(2.0).setNumHashTables(10).setInputCol("features").setOutputCol("hashes")
    val modelBrp = brp.fit(ratings)
    val hashedRatings = modelBrp.transform(ratings)

    // fitting
    val ifs = new com.github.takemikami.selica.ml.recommendation.ItemFeatureSimilarity()
      .setItemCol("itemId")
      .setFeaturesCol("hashes")
    val model = ifs.fit(hashedRatings)

  }

}
