import unittest
from pyspark.sql import SparkSession, SQLContext
from pyselica.ml.feature import JapaneseTokenizer


class TestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.spark = SparkSession.builder.appName('unittest').getOrCreate()

    def test_japanese_tokenizer(self):
        tokenizer = JapaneseTokenizer(inputCol="sentence", outputCol="words")

        sqlContext = SQLContext(self.spark)
        df = sqlContext.createDataFrame([(1, "すもももももももものうち")], ["id", "sentence"])
        tokenized = tokenizer.transform(df)

        pdf = tokenized.toPandas()
        self.assertTrue('words' in pdf.columns)
        self.assertEquals("すもももももももものうち", ''.join(pdf[pdf['id'] == 1].words.values[0]))
