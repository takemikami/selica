import unittest
from pyspark.sql import SparkSession, SQLContext
from pyselica.ml.pipeline import CallbackPipeline


class TestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.spark = SparkSession.builder.appName('unittest').getOrCreate()

    def test_callback_pipeline(self):
        # TODO
        pass
        # pipeline = CallbackPipeline(stages=[estimator1, estimator2],
        #                             callbacks=[None,
        #                                        lambda s, t: s.setBBB(t[0].aaaa)
        #                                        ])
