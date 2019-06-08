from pyspark import keyword_only
from pyspark.ml import Estimator, Transformer, Model
from pyspark.ml.wrapper import JavaTransformer
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import FloatType, ArrayType
import numpy as np


class JapaneseTokenizer(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(JapaneseTokenizer, self).__init__()
        self._java_obj = self._new_java_obj("com.github.takemikami.selica.ml.feature.JapaneseTokenizer", self.uid)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


class CosineSimilarityModel(Model):
    def __init__(self, vectorCol, similarityColPrefix, rankColPrefix, baseDataset):
        super(CosineSimilarityModel, self).__init__()
        self._vectorCol = vectorCol
        self._similarityColPrefix = similarityColPrefix
        self._rankColPrefix = rankColPrefix
        self._baseDataset = baseDataset

    def mostSimilar(self, vecs, k=None):
        cos_sim_udf = F.udf(lambda a, b: float(abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))),
                            FloatType())
        df = self._baseDataset
        for col_vec in zip(self._vectorCol, vecs):
            ary = F.array([F.lit(v) for v in col_vec[1]])
            sim_colname = "{}{}".format(self._similarityColPrefix, col_vec[0])
            rank_colname = "{}{}".format(self._rankColPrefix, col_vec[0])
            df = df.withColumn(sim_colname, cos_sim_udf(col_vec[0], ary)) \
                .withColumn(rank_colname, F.row_number().over(Window.orderBy(F.desc(sim_colname))))
            if k is not None:
                df = df.where(F.col(rank_colname) <= k)
        return df


class CosineSimilarity(Estimator):
    def __init__(self, vectorCol=["vector"], similarityColPrefix="sim_", rankColPrefix="rank_"):
        super(CosineSimilarity, self).__init__()
        self._vectorCol = vectorCol
        self._similarityColPrefix = similarityColPrefix
        self._rankColPrefix = rankColPrefix

    def _fit(self, dataset):
        return CosineSimilarityModel(self._vectorCol, self._similarityColPrefix, self._rankColPrefix, dataset)
