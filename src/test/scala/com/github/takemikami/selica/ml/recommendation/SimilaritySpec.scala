package com.github.takemikami.selica.ml.recommendation

import org.apache.spark.sql.SparkSession
import org.scalatest._
import Matchers._
import com.github.takemikami.selica.SparkSessionForUnitTest
import org.apache.spark.mllib.linalg.DenseMatrix

class SimilaritySpec extends FlatSpec with BeforeAndAfter {

  private var sparkSession: SparkSession = _
  before {
    sparkSession = SparkSessionForUnitTest.getSession()
  }
  after {
  }

  "CosineSimilarity" should "size is item x item" in {
    val spark = sparkSession
    import spark.implicits._

    val ratings = Seq(
      (0, 0, 1.0, 1L),
      (0, 1, 0.1, 1L),
      (0, 2, 0.1, 1L),
      (1, 0, 1.0, 1L),
      (1, 1, 0.5, 1L),
      (2, 1, 0.5, 1L),
      (3, 2, 0.5, 1L)
    ).toDF("userId", "itemId", "rating", "timestamp")

    val similarityMatrix = CosineSimilarity.train(ratings, "itemId", "userId", "rating")

    similarityMatrix.numCols() shouldEqual 3
    similarityMatrix.numRows() shouldEqual 3

    similarityMatrix.entries.foreach(println)
  }

  "CosineSimilarity" should "able to bruteforce compute" in {
    val spark = sparkSession
    import spark.implicits._

    val ratings = Seq(
      (0, 0, 1.0, 1L),
      (0, 1, 0.1, 1L),
      (0, 2, 0.1, 1L),
      (1, 0, 1.0, 1L),
      (1, 1, 0.5, 1L),
      (2, 1, 0.5, 1L),
      (3, 2, 0.5, 1L)
    ).toDF("userId", "itemId", "rating", "timestamp")

    val similarityMatrix = CosineSimilarity.train(ratings, "itemId", "userId", "rating", bruteforce = true)

    similarityMatrix.numCols() shouldEqual 3
    similarityMatrix.numRows() shouldEqual 3

    similarityMatrix.entries.foreach(println)
  }

}
