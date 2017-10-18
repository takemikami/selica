/*
 * Copyright (C) 2017 Takeshi Mikami.
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

import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors => OldVectors}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions.max
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

// Collaborative Filtering Model
private[recommendation] trait ItemBasedCollaborativeFilteringModelParams extends Params {
  val userCol = new Param[String](this, "userCol", "column name for user ids. Ids must be within the string value range.")
  val itemCol = new Param[String](this, "itemCol", "column name for item ids. Ids must be within the string value range.")
  val userIndexCol = new Param[String](this, "userIndexCol", "column index for user ids. model internal use.")
  val itemIndexCol = new Param[String](this, "itemIndexCol", "column index for item ids. model internal use.")
  val ratingCol = new Param[String](this, "ratingCol", "column name for ratings")
  val predictionCol = new Param[String](this, "predictionCol", "column name for predicted ratings")

  setDefault(userCol, "userId")
  setDefault(itemCol, "itemId")
  setDefault(userIndexCol, "userIndex")
  setDefault(itemIndexCol, "itemIndex")
  setDefault(ratingCol, "rating")
  setDefault(predictionCol, "prediction")
}

class ItemBasedCollaborativeFilteringModel (override val uid: String, val itemSimilarity: CoordinateMatrix, val itemIndex: StringIndexerModel)
  extends Model[ItemBasedCollaborativeFilteringModel]
    with ItemBasedCollaborativeFilteringModelParams {

  def setUserCol(value: String): this.type = set(userCol, value)
  def setItemCol(value: String): this.type = set(itemCol, value)
  def setUserIndexCol(value: String): this.type = set(userIndexCol, value)
  def setItemIndexCol(value: String): this.type = set(itemIndexCol, value)
  def setRatingCol(value: String): this.type = set(ratingCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  val similarityDenseMatrix: DenseMatrix = {
    val similarityMartixLocal = itemSimilarity.toBlockMatrix().toLocalMatrix()

    val similarityArray = similarityMartixLocal.toArray
    val similarityArrayTranspose = similarityMartixLocal.transpose.toArray
    val identityIdx = for(i <- 0 until similarityMartixLocal.numRows) yield i * similarityMartixLocal.numRows + i
    val doubleSimilarityArray = for(i <- 0 until similarityArray.length)
      yield if(identityIdx.exists(_==i)) 1.0 else similarityArray(i) + similarityArrayTranspose(i)
    new DenseMatrix(similarityMartixLocal.numRows, similarityMartixLocal.numCols, doubleSimilarityArray.toArray)
  }

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

  override def copy(extra: ParamMap): ItemBasedCollaborativeFilteringModel = {
    val copied = new ItemBasedCollaborativeFilteringModel(uid, itemSimilarity, itemIndex)
    copyValues(copied, extra).setParent(parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    // id -> index
    val userIndexer = new StringIndexer().setInputCol($(userCol)).setOutputCol($(userIndexCol))
    val userIndex = userIndexer.fit(dataset)
    val df = userIndex.transform(itemIndex.transform(dataset))

    // transform
    val baseColumnIdx = df.schema.fieldIndex($(itemIndexCol))
    val featureSampleColumnIdx = df.schema.fieldIndex($(userIndexCol))
    val scoreColumnIdx = df.schema.fieldIndex($(ratingCol))

    def returnInt(v: Any) = v match {
      case v: Double => v.toInt
      case v: Int => v
    }
    val baseSize = returnInt(df.agg((max($(itemIndexCol)))).head.get(0)) + 1

    val featureRdd = df.rdd.map{
      row => returnInt(row.get(featureSampleColumnIdx)) -> Seq((returnInt(row.get(baseColumnIdx)), row.getDouble(scoreColumnIdx)))
    }.reduceByKey((k, v) => k ++ v).map{
      v:(Int,Seq[(Int, Double)]) => (v._1, similarityDenseMatrix.multiply(OldVectors.fromML(Vectors.sparse(baseSize, v._2)).toDense))
    }.flatMap{
      row => {
        val vals = row._2.toArray
        Array.tabulate(vals.length){ i => (row._1, i, vals(i)) }
      }
    }

    val spark = SparkSession
      .builder
      .appName("SparkCF")
      .getOrCreate()
    import spark.implicits._
    val predictedDf = featureRdd.toDF($(userIndexCol), $(itemIndexCol), $(predictionCol))

    // index -> id
    val userIdColumn: Int => String = userIndex.labels(_)
    val itemIdColumn: Int => String = itemIndex.labels(_)
    val userIdUDF = udf(userIdColumn)
    val itemIdUDF = udf(itemIdColumn)

    val idbaseDf = predictedDf
      .withColumn($(userCol), userIdUDF(Symbol($(userIndexCol))))
      .withColumn($(itemCol), itemIdUDF(Symbol($(itemIndexCol))))
      .select($(userCol), $(itemCol), $(predictionCol))

    // join original rating
    idbaseDf
      .join(dataset, idbaseDf.col($(userCol)) === dataset.col($(userCol)) and idbaseDf.col($(itemCol)) === dataset.col($(itemCol)))
      .select(idbaseDf.col($(userCol)), idbaseDf.col($(itemCol)), idbaseDf.col($(predictionCol)), dataset.col($(ratingCol)))
  }

  override def transformSchema(schema: StructType): StructType = {
    require(schema($(userCol)).dataType.isInstanceOf[StringType], "invalid type: " + schema($(userCol)).dataType)
    require(schema($(itemCol)).dataType.isInstanceOf[StringType], "invalid type: " + schema($(itemCol)).dataType)
    require(schema($(ratingCol)).dataType.isInstanceOf[NumericType], "invalid type: " + schema($(ratingCol)).dataType)
    require(!schema.fieldNames.contains($(predictionCol)), s"already exists: ${userIndexCol}")
    require(!schema.fieldNames.contains($(predictionCol)), s"already exists: ${itemIndexCol}")
    require(!schema.fieldNames.contains($(predictionCol)), s"already exists: ${predictionCol}")
    StructType(schema.fields :+ StructField($(predictionCol), FloatType, false))
  }

}

// Collaborative Filtering Trainer
private[recommendation] trait ItemBasedCollaborativeFilteringParams extends ItemBasedCollaborativeFilteringModelParams {
}

class ItemBasedCollaborativeFiltering (override val uid: String)
  extends Estimator[ItemBasedCollaborativeFilteringModel]
    with ItemBasedCollaborativeFilteringParams {

  def this() = this(Identifiable.randomUID("collaborativefiltering"))

  def setUserCol(value: String): this.type = set(userCol, value)
  def setItemCol(value: String): this.type = set(itemCol, value)
  def setUserIndexCol(value: String): this.type = set(userIndexCol, value)
  def setItemIndexCol(value: String): this.type = set(itemIndexCol, value)
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  override def fit(dataset: Dataset[_]): ItemBasedCollaborativeFilteringModel = {
    val itemIndexer = new StringIndexer().setInputCol($(itemCol)).setOutputCol($(itemIndexCol))
    val userIndexer = new StringIndexer().setInputCol($(userCol)).setOutputCol($(userIndexCol))
    val itemIndex = itemIndexer.fit(dataset)
    val userIndex = userIndexer.fit(dataset)
    val df = userIndex.transform(itemIndex.transform(dataset))

    val similarityMatrix = CosineSimilarity.train(df, $(itemIndexCol), $(userIndexCol), $(ratingCol))

    val model = new ItemBasedCollaborativeFilteringModel(uid, similarityMatrix, itemIndex)
    copyValues(model)
  }

  override def copy(extra: ParamMap):ItemBasedCollaborativeFiltering = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    require(schema($(userCol)).dataType.isInstanceOf[StringType], "invalid type: " + schema($(userCol)).dataType)
    require(schema($(itemCol)).dataType.isInstanceOf[StringType], "invalid type: " + schema($(itemCol)).dataType)
    require(schema($(ratingCol)).dataType.isInstanceOf[NumericType], "invalid type: " + schema($(ratingCol)).dataType)
    StructType(schema.fields)
  }

}
