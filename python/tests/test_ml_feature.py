import unittest

from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession, SQLContext
from pyselica.ml.feature import JapaneseTokenizer, CosineSimilarity


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

    def test_cosine_similarity(self):
        sqlContext = SQLContext(self.spark)
        df = sqlContext.createDataFrame([
            (1, Vectors.dense([1.0, 1.0])),
            (2, Vectors.dense([1.0, 0.9])),
            (3, Vectors.dense([1.0, 0.5])),
            (4, Vectors.dense([0.5, 1.0])),
        ], ["id", "vector"])

        cossim = CosineSimilarity().fit(df)
        rtn = cossim.mostSimilar([Vectors.dense([1.0, 1.0])])

        pdf = rtn.toPandas()
        self.assertTrue('sim_vector' in pdf.columns)
        self.assertTrue('rank_vector' in pdf.columns)
        self.assertEquals(1, pdf[pdf['rank_vector'] == 1].id.values[0])

    def test_cosine_similarity_multi(self):
        sqlContext = SQLContext(self.spark)
        df = sqlContext.createDataFrame([
            (1, Vectors.dense([1.0, 1.0]), Vectors.dense([1.0, 0.9])),
            (2, Vectors.dense([1.0, 0.9]), Vectors.dense([0.5, 1.0])),
            (3, Vectors.dense([1.0, 0.5]), Vectors.dense([1.0, 0.5])),
            (4, Vectors.dense([0.5, 1.0]), Vectors.dense([1.0, 1.0])),
        ], ["id", "vector1", "vector2"])

        cossim = CosineSimilarity(vectorCol=["vector1", "vector2"]).fit(df)
        rtn = cossim.mostSimilar([Vectors.dense([1.0, 1.0]), Vectors.dense([1.0, 1.0])])
        
        pdf = rtn.toPandas()
        self.assertTrue('sim_vector1' in pdf.columns)
        self.assertTrue('rank_vector1' in pdf.columns)
        self.assertEquals(1, pdf[pdf['rank_vector1'] == 1].id.values[0])
        self.assertEquals(4, pdf[pdf['rank_vector2'] == 1].id.values[0])
