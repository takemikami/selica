package com.github.takemikami.selica.ml.recommendation

import com.github.takemikami.selica.ml.fpm.FrequentViewConversionPatternMiningModel
import com.github.takemikami.selica.ml.udf.IndexSumFunction
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

// Attribute Clustered Recommender Model
private[recommendation] trait AttributeClusteredRecommenderModelParams extends Params {
  val attributeCol = new Param[String](this, "attributeCol", "column name for attribute id. Ids must be within the string value.")

  setDefault(
    attributeCol -> "attribute"
  )
}

class AttributeClusteredRecommenderModel (override val uid: String, val clusterMap: DataFrame, val ruleDataFrame: DataFrame)
  extends Model[AttributeClusteredRecommenderModel]
    with AttributeClusteredRecommenderModelParams {

  override def copy(extra: ParamMap): AttributeClusteredRecommenderModel = {
    val copied = new AttributeClusteredRecommenderModel(uid, clusterMap, ruleDataFrame)
    copyValues(copied, extra).setParent(parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = ???

  override def transformSchema(schema: StructType): StructType = ???
}

// Attribute Clustered Recommender Trainer
private[recommendation] trait AttributeClusteredRecommenderParams extends AttributeClusteredRecommenderModelParams {
  val itemCol = new Param[String](this, "itemCol", "column name for item id string. Id must be within the int value.")
  val minSupport = new DoubleParam(this, "minSupport", "minimal support.")
  val minConfidence = new DoubleParam(this, "minConfidence", "minimal confidence.")
  val kMeansK = new IntParam(this, "kMeansK", "KMeans parameter K.")
  val kMeansSeed = new LongParam(this, "kMeansSeed", "KMeans parameter Seed.")

  setDefault(
    itemCol -> "itemid",
    minSupport -> 0.001,
    minConfidence -> 0.001,
    kMeansK -> 5,
    kMeansSeed -> 1L
  )
}

class AttributeClusteredRecommender (override val uid: String)
  extends Estimator[AttributeClusteredRecommenderModel]
    with AttributeClusteredRecommenderParams {

  def this() = this(Identifiable.randomUID("attributeclusteredrecommender"))

  def setAttributeCol(value: String): this.type = set(attributeCol, value)
  def setItemCol(value: String): this.type = set(itemCol, value)
  def setMinSupport(value: Double): this.type = set(minSupport, value)
  def setKMeansK(value: Int): this.type = set(kMeansK, value)
  def setKMeansSeed(value: Long): this.type = set(kMeansSeed, value)

  override def fit(dataset: Dataset[_]): AttributeClusteredRecommenderModel = {
    import dataset.sparkSession.sqlContext.implicits._

    // indexing item & attribute
    val itemIndex = new StringIndexer().setInputCol($(itemCol)).setOutputCol("itemIndex").fit(dataset)
    val attrIndex = new StringIndexer().setInputCol($(attributeCol)).setOutputCol("attrIndex").fit(dataset)
    val indexedDataset = attrIndex.transform(itemIndex.transform(dataset))

    // clustering
    val itemCnt = new IndexSumFunction
    val itemSize = udf(() => itemIndex.labels.length)
    val itemFeatureDataset = indexedDataset.groupBy('attrIndex)
      .agg(itemCnt('itemIndex, itemSize()) as "features")

    val kmeans = new KMeans().setK($(kMeansK)).setSeed($(kMeansSeed)).setPredictionCol("attrClass")
    val clusterModel = kmeans.fit(itemFeatureDataset)
    val clusterdDataset = clusterModel.transform(itemFeatureDataset)

    val clusterMap = clusterdDataset.select('attrIndex, 'attrClass).rdd
      .map { x => x(0) -> x(1) }
      .map { case (x: Double, y: Int) => x.toInt -> y.toInt }
      .collect().toMap

    // Support
    val numOfTransaction = indexedDataset.count()

    // compute antecedent supports
    val vectorValueSum = udf((vec: SparseVector) => vec.toArray.toList.sum)
    val antecedentSupportMap = clusterdDataset.select('attrClass, vectorValueSum('features)).rdd
      .map(x => x(0) -> x(1))
      .map{ case (x: Int, y: Double) => x.toInt -> y.toInt }
      .filter { case (_: Int, y: Int) => (y.toDouble / numOfTransaction) > $(minSupport) }
      .collect().toMap
    val antecedentItems = antecedentSupportMap.keySet

    // compute consequent supports
    val consequenceSupportMap = indexedDataset.groupBy('itemIndex).count().rdd
      .map(x => x(0) -> x(1))
      .map{ case (x: Double, y: Long) => x.toInt -> y.toInt }
      .filter { case (_: Any, y: Int) => (y.toDouble / numOfTransaction) > $(minSupport) }
      .collect().toMap
    val consequenceItems = consequenceSupportMap.keySet

    // compute Support, Confidence, Lift
    val clusterIdx = udf((attrIdx: Double) => clusterMap(attrIdx.toInt))
    val itemId = udf((itemIdx: Int) => itemIndex.labels(itemIdx))
    val ruleDF = indexedDataset.select(clusterIdx('attrIndex), 'itemIndex).rdd
      .map(x => x(0) -> x(1))
      .map{ case (x: Int, y: Double) => x.toInt -> y.toInt }
      .filter {
        case (x: Int, y: Int) => antecedentItems.contains(x) && consequenceItems.contains(y)
      }
      .map{ x => (x, 1) }
      .reduceByKey(_ + _)
      .map {
        case ((i: Int, j: Int), c: Int) => (
          i, // antecedent (item_i)
          j, // consequent (item_j)
          c.toDouble / numOfTransaction, // support
          c.toDouble / antecedentSupportMap(i), // confidence
          (c.toDouble / antecedentSupportMap(i)) / (consequenceSupportMap(j).toDouble / numOfTransaction) // lift
        )
      }
      .toDF("attrClass", "itemIndex", "support", "confidence", "lift")
      .select('attrClass, itemId('itemIndex) as "itemId", 'support, 'confidence, 'lift)

    val clusterMapDf = indexedDataset
      .select($(attributeCol), "attrIndex")
      .distinct()
      .select(Symbol($(attributeCol)), clusterIdx('attrIndex) as "attrClass")

    val model = new AttributeClusteredRecommenderModel(uid, clusterMapDf, ruleDF)
    copyValues(model)
  }

  override def copy(extra: ParamMap): Estimator[AttributeClusteredRecommenderModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    require(schema($(attributeCol)).dataType.isInstanceOf[String], "invalid type: " + schema($(attributeCol)).dataType)
    StructType(schema.fields)
  }

}
