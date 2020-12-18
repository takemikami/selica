package com.github.takemikami.selica.ml.udf

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

// アイテム数えて特徴量のベクトルにするやつUDAF
class IndexSumFunction extends UserDefinedAggregateFunction  {
  override def inputSchema: StructType = StructType(
    StructField("value", IntegerType)
      :: StructField("featureSize", IntegerType)
      :: Nil)

  override def bufferSchema: StructType = StructType(
    StructField("map", MapType(IntegerType, IntegerType))
      :: StructField("size", IntegerType)
      :: Nil
  )

  override def dataType: DataType = org.apache.spark.ml.linalg.SQLDataTypes.VectorType

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = mutable.Map[Int, Int]()
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val idx = input.getAs[Int](0)
    val map = buffer.getAs[Map[Int, Int]](0)

    buffer(0) = map + (idx -> { map.getOrElse(idx, 0) + 1 })
    buffer(1) = input.getAs[Int](1)
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    var map = mutable.Map[Int, Int]()
    val mapL = buffer1.getAs[Map[Int, Int]](0)
    val mapR = buffer2.getAs[Map[Int, Int]](0)
    (mapL.keySet ++ mapR.keySet).foreach(k =>
      map += (k -> {mapL.getOrElse(k, 0) + mapR.getOrElse(k, 0)})
    )

    buffer1(0) = map
    buffer1(1) = buffer2.getAs[Int](1)
  }

  override def evaluate(buffer: Row): Any = {
    val map = buffer.getAs[Map[Int, Int]](0)
    var keyBuffer = ListBuffer[Int]()
    var valueBuffer = ListBuffer[Double]()
    //map.foreach(print)
    map.keySet.toList.sorted.foreach(k => {
      keyBuffer += k
      valueBuffer += map.get(k).get
    })

    val size = buffer.getAs[Int](1) //size
    Vectors.sparse(size, keyBuffer.toArray, valueBuffer.toArray)
  }
}
