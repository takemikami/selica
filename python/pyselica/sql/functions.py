from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
import numpy as np

cos_sim = F.udf(
    lambda a, b: float(abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))),
    FloatType())

centroid = F.udf(
    lambda v: np.mean(v, axis=0).tolist(),
    ArrayType(FloatType()))
