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

import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.linalg.distributed.{RowMatrix => OldRawMatrix}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.stat.Correlation

trait Similarity {
  def train(dataset: DataFrame, baseColumn: String, featureSampleColumn: String, scoreColumn: String): CoordinateMatrix
}

object CorrelationSimilariy { // extends Similarity {
  def train(dataset: DataFrame, baseColumn: String, featureSampleColumn: String, scoreColumn: String): CoordinateMatrix = {
    val baseColumnIdx = dataset.schema.fieldIndex(baseColumn)
    val featureSampleColumnIdx = dataset.schema.fieldIndex(featureSampleColumn)
    val scoreColumnIdx = dataset.schema.fieldIndex(scoreColumn)
    val baseSize = dataset.agg((max(baseColumn))).head.getInt(0) + 1

    // TODO: implements
    //    val df2 = dataset.rdd.map{
    //      row => row.getInt(featureSampleColumnIdx) -> Seq((row.getInt(baseColumnIdx), row.getDouble(scoreColumnIdx)))
    //    }.reduceByKey((k, v) => k ++ v).map{
    //      v:(Int,Seq[(Int, Double)]) => Vectors.sparse(baseSize, v._2)
    //    }.map(Tuple1.apply).toDF("features")

    new CoordinateMatrix(null)
  }
}
