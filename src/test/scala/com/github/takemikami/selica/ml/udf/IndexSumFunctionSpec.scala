package com.github.takemikami.selica.ml.udf


import com.github.takemikami.selica.SparkSessionForUnitTest
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import org.apache.spark.ml.linalg.{SparseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructField, StructType}
import Matchers._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.functions.udf

class IndexSumFunctionSpec extends FlatSpec with BeforeAndAfter  {
  private var sparkSession: SparkSession = _
  before {
    sparkSession = SparkSessionForUnitTest.getSession()
  }
  after {
  }

  "IndexCount" should "can count" in {
    val spark = sparkSession
    import spark.implicits._

    val itemDf = Seq(
      ("c0", 1, 1),
      ("c0", 2, 1),
      ("c0", 3, 1),
      ("c1", 4, 1),
      ("c1", 3, 1),
      ("c1", 2, 1),
      ("c2", 1, 1),
      ("c2", 0, 1),
      ("c2", 1, 1)
    ).toDF("clusterId", "itemIndex", "count")

    val itemCnt = new IndexSumFunction
    val itemSize = udf(() => 5)

    val rtn = itemDf.groupBy('clusterId)
      .agg(itemCnt('itemIndex, itemSize()) as "features")
    rtn.show()

    rtn.count() shouldEqual 3

    val vecC0 = rtn.select('features).where("clusterId = 'c0'").collect()(0).getAs[SparseVector](0)
    vecC0.indices shouldEqual Array(1, 2, 3)
    vecC0.values shouldEqual Array(1.0, 1.0, 1.0)
  }

  "IndexCountForClustering" should "prepare data for clustering" in {
    val spark = sparkSession
    import spark.implicits._

    val itemDf = Seq(
      ("u0", 1, 1),
      ("u1", 1, 1),
      ("u1", 2, 1),
      ("u2", 3, 1),
      ("u2", 4, 1),
      ("u3", 3, 1),
      ("u3", 4, 1),
      ("u4", 0, 1),
      ("u4", 3, 1),
      ("u5", 0, 1),
      ("u5", 4, 1),
      ("u6", 0, 1)
    ).toDF("userId", "itemIndex", "count")

    val itemCnt = new IndexSumFunction
    val itemSize = udf(() => 5)

    val df = itemDf.groupBy('userId)
      .agg(itemCnt('itemIndex, itemSize()) as "features")
    df.show()

    // clustering
    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(df)
    val predictions = model.transform(df)
    predictions.show()
  }
}
