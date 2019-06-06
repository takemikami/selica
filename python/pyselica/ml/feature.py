from pyspark import keyword_only
from pyspark.ml.wrapper import JavaTransformer
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.param.shared import HasInputCol, HasOutputCol


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
