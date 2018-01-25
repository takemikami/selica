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

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, RowMatrix => OldRawMatrix}
import org.apache.spark.sql.functions._

private[recommendation] trait ItemSimilarityModel {
}

private[recommendation] trait ItemSimilarity {
}

trait Similarity {
  def train(dataset: DataFrame, baseColumn: String, featureSampleColumn: String, scoreColumn: String): CoordinateMatrix
}

object CosineSimilarity {
  def train(
             dataset: DataFrame,
             baseColumn: String,
             featureSampleColumn: String,
             scoreColumn: String,
             threshold: Double = 0.1,
             bruteforce: Boolean = false
           ): CoordinateMatrix = {

    val baseColumnIdx = dataset.schema.fieldIndex(baseColumn)
    val featureSampleColumnIdx = dataset.schema.fieldIndex(featureSampleColumn)
    val scoreColumnIdx = dataset.schema.fieldIndex(scoreColumn)

    def returnInt(v: Any) = v match {
      case v: Double => v.toInt
      case v: Int => v
    }
    val baseSize = returnInt(dataset.agg((max(baseColumn))).head.get(0)) + 1

    val featureRdd = dataset.rdd.map{
      row => returnInt(row.get(featureSampleColumnIdx)) -> Seq((returnInt(row.get(baseColumnIdx)), row.getDouble(scoreColumnIdx)))
    }.reduceByKey((k, v) => k ++ v).map{
      v:(Int,Seq[(Int, Double)]) => OldVectors.fromML(Vectors.sparse(baseSize, v._2))
    }
    val mat = new OldRawMatrix(featureRdd)

    if (bruteforce) {
      mat.columnSimilarities() // brute force compute
    } else {
      mat.columnSimilarities(threshold) // DIMSUM
    }
  }
}
