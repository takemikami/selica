import unittest
from pyspark.sql import SparkSession, SQLContext
from pyselica.ml.evaluation import ClusteringDiffEvaluator


class TestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.spark = SparkSession.builder.appName('unittest').getOrCreate()

    def test_clustering_diff_evaluator(self):
        sqlContext = SQLContext(self.spark)
        base_predictions = sqlContext.createDataFrame([(1, 1), (2, 1), (3, 1), (4, 1)], ["id", "prediction"])
        predictions1 = sqlContext.createDataFrame([(1, 1), (2, 1), (3, 1), (4, 1)], ["id", "prediction"])
        predictions2 = sqlContext.createDataFrame([(1, 1), (2, 1), (3, 1), (4, 0)], ["id", "prediction"])

        evaluator = ClusteringDiffEvaluator(base_predictions)
        diff_rate1 = evaluator.evaluate(predictions1)
        self.assertEqual(1.0, diff_rate1)

        diff_rate2 = evaluator.evaluate(predictions2)
        self.assertTrue(diff_rate2 > 0 and diff_rate2 < 1.0)
        self.assertEqual(0.7692307692307692, diff_rate2)
