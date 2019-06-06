from pyspark.ml.evaluation import Evaluator
from pyspark.sql import functions as F


class ClusteringDiffEvaluator(Evaluator):
    def __init__(self, baseDataset, baseIdCol="id", basePredictionCol="prediction",
                 idCol="id", predictionCol="prediction", sampleSize=None, metricName="f-value"):
        super(ClusteringDiffEvaluator, self).__init__()
        self._baseDataset = baseDataset
        self._baseIdCol = baseIdCol
        self._idCol = idCol
        self._basePredictionCol = basePredictionCol
        self._predictionCol = predictionCol
        self._sampleSize = sampleSize
        self._metricName = metricName

    def _evaluate(self, dataset):
        # join base prediction and dataset
        joined = self._baseDataset.alias("actual") \
            .join(dataset.alias("pred"),
                  F.col("actual.{}".format(self._baseIdCol)) == F.col("pred.{}".format(self._idCol))) \
            .select(F.col("actual.{}".format(self._basePredictionCol)).alias("aclust"),
                    F.col("pred.{}".format(self._predictionCol)).alias("pclust"))

        # sampling
        if self._sampleSize is not None:
            joined = joined.sample(True, self._sampleSize)
        joined.cache().collect()

        df_cf = joined.alias("left").crossJoin(joined.alias("right")) \
            .withColumn("aset", F.col("left.pclust") == F.col("right.pclust")) \
            .withColumn("pset", F.col("left.aclust") == F.col("right.aclust")) \
            .groupby("aset", "pset").agg(F.count("*").alias("cnt"))
        cf = df_cf.toPandas()

        # compute metrics
        from scipy import stats
        cnt_pp = cf[(cf.aset == True) & (cf.pset == True)].cnt.sum()
        cnt_pn = cf[(cf.aset == True) & (cf.pset == False)].cnt.sum()
        cnt_np = cf[(cf.aset == False) & (cf.pset == True)].cnt.sum()
        precision = cnt_pp / (cnt_pp + cnt_pn)
        recall = cnt_pp / (cnt_pp + cnt_np)
        f_value = stats.hmean([precision, recall])

        if self._metricName == "precision":
            return precision
        if self._metricName == "recall":
            return recall
        return f_value
