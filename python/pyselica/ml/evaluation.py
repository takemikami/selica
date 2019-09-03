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
        # count pair size
        basePositive = self._baseDataset.alias("left") \
            .join(self._baseDataset.alias("right"), self._basePredictionCol) \
            .select(F.col("left.{}".format(self._baseIdCol)).alias("id1"),
                    F.col("right.{}".format(self._baseIdCol)).alias("id2"))
        predPositive = dataset.alias("left") \
            .join(dataset.alias("right"), self._predictionCol) \
            .select(F.col("left.{}".format(self._baseIdCol)).alias("id1"),
                    F.col("right.{}".format(self._idCol)).alias("id2"))

        # join base prediction and dataset
        truePositive = basePositive.join(predPositive, ["id1", "id2"])

        # compute metrics
        from scipy import stats
        basePositiveSize = basePositive.count()
        predPositiveSize = predPositive.count()
        truePositiveSize = truePositive.count()
        precision = truePositiveSize / predPositiveSize
        recall = truePositiveSize / basePositiveSize
        f_value = stats.hmean([precision, recall])

        if self._metricName == "precision":
            return precision
        if self._metricName == "recall":
            return recall
        return f_value
