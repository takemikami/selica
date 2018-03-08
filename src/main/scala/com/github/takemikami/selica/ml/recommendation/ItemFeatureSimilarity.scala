/*
 * Copyright (C) 2018 Takeshi Mikami.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.takemikami.selica.ml.recommendation

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors => OldVectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, RowMatrix => OldRawMatrix}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{udf, collect_list, map, col}

// Item Feature Similarity Model
private[recommendation] trait ItemFeatureSimilarityModelParams extends Params {
  val itemCol = new Param[String](this, "itemCol", "column name for item ids. Ids must be within the string value range.")
  val nearestItemsCol = new Param[String](this, "nearestItemCol", "column name for nearest items.")

  setDefault(
    itemCol -> "itemId",
    nearestItemsCol -> "nearestItems"
  )
}

class ItemFeatureSimilarityModel (override val uid: String, val itemSimilarity: CoordinateMatrix, val itemIndex: StringIndexerModel)
  extends Model[ItemFeatureSimilarityModel]
    with ItemFeatureSimilarityModelParams {

  def setItemCol(value: String): this.type = set(itemCol, value)
  def setNearestItemCol(value: String): this.type = set(nearestItemsCol, value)

  lazy val similarityDataFrame: DataFrame = {
    val itemIdColumn: Int => String = itemIndex.labels(_)
    val itemIdUDF = udf(itemIdColumn)

    val spark = SparkSession
      .builder
      .appName("SparkCF")
      .getOrCreate()
    import spark.implicits._
    val entries = itemSimilarity.entries.map(e => (e.i, e.j, e.value)).toDF("item_i_index", "item_j_index", "similarity")

    entries
      .withColumn("item_i", itemIdUDF('item_i_index))
      .withColumn("item_j", itemIdUDF('item_j_index))
      .select("item_i", "item_j", "similarity")
  }
  override def copy(extra: ParamMap): ItemFeatureSimilarityModel = {
    val copied = new ItemFeatureSimilarityModel(uid, itemSimilarity, itemIndex)
    copyValues(copied, extra).setParent(parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val listMapFlat = udf { values: Seq[Map[String,Double]] => values.flatten.toMap }

    val crossSimilarityDf = similarityDataFrame.withColumnRenamed("item_i", "item_x").withColumnRenamed("item_j", "item_y")
      .union(similarityDataFrame.withColumnRenamed("item_j", "item_x").withColumnRenamed("item_i", "item_y"))
    val joinedDf = dataset.join(crossSimilarityDf, dataset($(itemCol)) === crossSimilarityDf("item_y")).select($(itemCol), "item_x", "similarity")

    joinedDf
      .groupBy($(itemCol))
      .agg(collect_list(map(col("item_x"), col("similarity"))) as "listmap")
      .withColumn($(nearestItemsCol), listMapFlat(col("listmap")))
      .drop("listmap")
  }

  override def transformSchema(schema: StructType): StructType = {
    require(schema($(itemCol)).dataType.isInstanceOf[StringType], "invalid type: " + schema($(itemCol)).dataType)
    require(!schema.fieldNames.contains($(nearestItemsCol)), s"already exists: ${nearestItemsCol}")
    StructType(schema.fields :+ StructField($(nearestItemsCol), MapType(StringType, FloatType), false))
  }

}

// Item Feature Similarity Trainer
private[recommendation] trait ItemFeatureSimilarityParams extends ItemFeatureSimilarityModelParams {
  val bruteForce = new BooleanParam(this, "bruteForce", "Compute similar columns perfectly, with brute force. (DIMSUM if false)")
  val threshold = new DoubleParam(this, "threshold", "dimsum threshold.")
  val itemIndexCol = new Param[String](this, "itemIndexCol", "column index for item ids. model internal use.")
  val featuresCol = new Param[String](this, "featuresCol", "column name for features. (SparseVector or WrappedArray)")

  setDefault(
    bruteForce -> false,
    itemIndexCol -> "itemIndex",
    featuresCol -> "fetures",
    threshold -> 0.1
  )
}

class ItemFeatureSimilarity (override val uid: String)
  extends Estimator[ItemFeatureSimilarityModel]
    with ItemFeatureSimilarityParams {

  def this() = this(Identifiable.randomUID("itemfeaturesimilarity"))

  def setItemCol(value: String): this.type = set(itemCol, value)
  def setItemIndexCol(value: String): this.type = set(itemIndexCol, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setBruteForce(value: Boolean): this.type = set(bruteForce, value)
  def setThreshold(value: Double): this.type = set(threshold, value)

  override def fit(dataset: Dataset[_]): ItemFeatureSimilarityModel = {
    val itemIndexer = new StringIndexer().setInputCol($(itemCol)).setOutputCol($(itemIndexCol))
    val itemIndex = itemIndexer.fit(dataset)
    val df = itemIndex.transform(dataset)
    val itemsize = df.count().toInt

    val featureRdd = df.select($(itemIndexCol), $(featuresCol)).rdd.map { row =>
      val itemIdx = row(0) match {case d:Double => d.toInt }
      row(1) match {
        case vec: SparseVector =>
          vec.indices.zipWithIndex.map { case (vi, num) => (vi, itemIdx, vec.values(num)) }.toList
        case vec: DenseVector =>
          vec.toArray.zipWithIndex.map { case (v, num) => (num, itemIdx, v) }.toList
        case ary: scala.collection.mutable.WrappedArray[org.apache.spark.ml.linalg.DenseVector] =>
          ary.zipWithIndex.map { case (vv, vi) => (vi, itemIdx, vv(0)) }.toList
      }
    }.flatMap(f => f).map(e => e._1 -> Seq((e._2, e._3))).reduceByKey((k, v) => k ++ v).map {
      case (_: Int, v: Seq[(Int, Double)]) => OldVectors.fromML(Vectors.sparse(itemsize, v))
    }
    val mat = new OldRawMatrix(featureRdd)

    // compute by brute force(=0.0) or DIMSUM
    val similarityMatrix = mat.columnSimilarities(if ($(bruteForce)) 0.0 else $(threshold))

    val model = new ItemFeatureSimilarityModel(uid, similarityMatrix, itemIndex)
    copyValues(model)
  }

  override def copy(extra: ParamMap): ItemFeatureSimilarity = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    require(schema($(itemCol)).dataType.isInstanceOf[StringType], "invalid type: " + schema($(itemCol)).dataType)
    if (schema($(featuresCol)).dataType != org.apache.spark.ml.linalg.SQLDataTypes.VectorType) {
      throw new IllegalArgumentException(s"invalid type: " + schema($(featuresCol)).dataType)
    }
    StructType(schema.fields)
  }
}
