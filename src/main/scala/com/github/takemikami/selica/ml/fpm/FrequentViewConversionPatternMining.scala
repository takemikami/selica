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
package com.github.takemikami.selica.ml.fpm

import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{row_number, udf, broadcast}

// Frequent View-Conversion PatternMining Model
private[fpm] trait FrequentViewConversionPatternMiningModelParams extends Params {
  val antecedentCol = new Param[String](this, "antecedentCol", "column name for antecedent id array. Ids must be within the int value.")
  val predictionCol = new Param[String](this, "predictionCol", "column name for predicted consequent id array. Ids must be within the int value.")

  setDefault(
    antecedentCol -> "antecedent",
    predictionCol -> "prediction"
  )
}

class FrequentViewConversionPatternMiningModel(override val uid: String, val ruleDataFrame: DataFrame)
  extends Model[FrequentViewConversionPatternMiningModel]
    with FrequentViewConversionPatternMiningModelParams {

  override def copy(extra: ParamMap): FrequentViewConversionPatternMiningModel = {
    val copied = new FrequentViewConversionPatternMiningModel(uid, ruleDataFrame)
    copyValues(copied, extra).setParent(parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.sqlContext.implicits._

    // make antesedent -> top consequent map
    val w = Window.partitionBy($"item_i").orderBy($"confidence".desc) //, //$"lift".desc)
    val topConsequent = ruleDataFrame.withColumn("rn", row_number.over(w)).where($"rn" === 1).drop("rn")
    val topConsequentMap = topConsequent.select("item_i", "item_j").rdd.map(r => r(0) -> r(1)).collect().toMap

    // define udf
    val consequent: scala.collection.mutable.WrappedArray[Int] => Option[Int] = { x =>
      val consList = x.map(topConsequentMap.get(_)).toList
      val consCandidates = consList.groupBy(identity).map{ case (k, v) => (k, v.size) }
      val consMax = consCandidates.filter{ case (k, v) => v == consCandidates.values.max }.keys.toList
      consList.reverse.filter(consMax.contains(_)).headOption.getOrElse(None) match {
        case Some(x: Int) => Option(x)
        case _ => None
      }
    }
    val consequentUDF = udf(consequent)

    // compute top consequent
    dataset.withColumn($(predictionCol), consequentUDF(Symbol($(antecedentCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    require(schema($(antecedentCol)).dataType.isInstanceOf[ArrayType], "invalid type: " + schema($(antecedentCol)).dataType)
    require(!schema.fieldNames.contains($(predictionCol)), s"already exists: ${predictionCol}")
    StructType(schema.fields :+ StructField($(predictionCol), IntegerType, false))
  }
}

// Frequent View-Conversion PatternMining Trainer
private[fpm] trait FrequentViewConversionPatternMiningParams extends FrequentViewConversionPatternMiningModelParams {
  val consequentCol = new Param[String](this, "consequentCol", "column name for consequent id array. Ids must be within the int value.")
  val minSupport = new DoubleParam(this, "minSupport", "minimal support.")
  val minConfidence = new DoubleParam(this, "minConfidence", "minimal confidence.")

  setDefault(
    consequentCol -> "consequent",
    minSupport -> 0.001,
    minConfidence -> 0.001
  )
}

class FrequentViewConversionPatternMining(override val uid: String)
  extends Estimator[FrequentViewConversionPatternMiningModel]
    with FrequentViewConversionPatternMiningParams {

  def this() = this(Identifiable.randomUID("fvcpm"))

  def setAntecedentCol(value: String): this.type = set(antecedentCol, value)

  def setConsequentCol(value: String): this.type = set(consequentCol, value)

  def setMinSupport(value: Double): this.type = set(minSupport, value)

  def setMinConfidence(value: Double): this.type = set(minConfidence, value)

  def supportMap(dataset: Dataset[_], targetCol: String, numOfTransaction: Long, minimalSupport: Double): Map[Int, Int] = {
    dataset.select(targetCol).rdd.map { t =>
      t(0) match {
        case x: scala.collection.mutable.WrappedArray[Int] => x.toSet
        case _ => List()
      }
    }.flatMap(f => f).map { x => (x, 1) }.reduceByKey(_ + _).filter {
      case (x: Int, y: Int) => (y.toDouble / numOfTransaction) > minimalSupport
    }.map {
      case (x: Int, y: Int) => x -> y
    }.collect().toMap
  }

  override def fit(dataset: Dataset[_]): FrequentViewConversionPatternMiningModel = {
    import dataset.sparkSession.sqlContext.implicits._

    // count transaction
    val numOfTransaction = dataset.count()

    // compute antecedent supports
    val antecedentSupportMap = supportMap(dataset, $(antecedentCol), numOfTransaction, $(minSupport))
    val antecedentItems = antecedentSupportMap.keySet

    // compute consequent supports
    val consequenceSupportMap = supportMap(dataset, $(consequentCol), numOfTransaction, $(minSupport))
    val consequenceItems = consequenceSupportMap.keySet

    // compute confidence
    val ruleDF = dataset.select("antecedent", "consequent").rdd.map { t =>
      (t(0), t(1)) match {
        case (x: scala.collection.mutable.WrappedArray[Int], y: scala.collection.mutable.WrappedArray[Int]) => {
          val ac = x.map(a => if (antecedentItems.contains(a)) a else 0).filter {
            _ != 0
          }
          val cs = y.map(a => if (consequenceItems.contains(a)) a else 0).filter {
            _ != 0
          }
          ac.map(i => cs.map(j => (i, j))).flatten.toList
        }
      }
    }.flatMap(f => f).map { x => (x, 1) }.reduceByKey(_ + _).filter {
      case ((i: Int, j: Int), c: Int) => (c.toDouble / numOfTransaction) > $(minSupport)
    }.map {
      case ((i: Int, j: Int), c: Int) => (
        i, // antecedent (item_i)
        j, // consequent (item_j)
        c.toDouble / numOfTransaction, // support
        c.toDouble / antecedentSupportMap(i), // confidence
        (c.toDouble / antecedentSupportMap(i)) / (consequenceSupportMap(j).toDouble / numOfTransaction) // lift
      )
    }.toDF("item_i", "item_j", "support", "confidence", "lift")

    val model = new FrequentViewConversionPatternMiningModel(uid, ruleDF)
    copyValues(model)
  }

  override def copy(extra: ParamMap): Estimator[FrequentViewConversionPatternMiningModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    require(schema($(antecedentCol)).dataType.isInstanceOf[ArrayType], "invalid type: " + schema($(antecedentCol)).dataType)
    require(schema($(consequentCol)).dataType.isInstanceOf[ArrayType], "invalid type: " + schema($(consequentCol)).dataType)
    StructType(schema.fields)
  }
}
