package com.github.takemikami.selica.ml.recommendation

import java.io.Serializable

import breeze.linalg.{max, sum}
import com.github.takemikami.selica.SparkSessionForUnitTest
import com.github.takemikami.selica.ml.udf.IndexSumFunction
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{StructField, UserDefinedType, _}
import org.scalatest.{BeforeAndAfter, FlatSpec}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

class AttributeClusteredRecommenderSpec extends FlatSpec with BeforeAndAfter {

  private var sparkSession: SparkSession = _
  before {
    sparkSession = SparkSessionForUnitTest.getSession()
  }
  after {
  }

  "AttributeClusteredRecommender" should "can fit & access model" in {
    val spark = sparkSession
    import spark.implicits._

    val ratings = Seq(
      ("Sun", "PC", "i1", 1),
      ("Sun", "SP", "i1", 1),
      ("Sun", "SP", "i1", 1),
      ("Mon", "SP", "i0", 1),
      ("Mon", "SP", "i0", 1),
      ("Mon", "SP", "i0", 1),
      ("Tue", "SP", "i1", 1),
      ("Tue", "SP", "i0", 1),
      ("Tue", "SP", "i0", 1)
    ).toDF("attr1", "attr2", "itemId", "quantity")

    val acr = new AttributeClusteredRecommender().setAttributeCol("attr1").setItemCol("itemId").setKMeansK(2)
    val acrModel = acr.fit(ratings)

    acrModel.clusterMap.show()
    acrModel.ruleDataFrame.show()
  }

}
