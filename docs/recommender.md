# Recommender

Recommender is information filtering algorithms for personalized item-recommend to each user.

## Item-based Collaborative filtering

Collaborative filtering is mainly use for recommender.
selica provide Item-based Collaborative filtering by cosine similarity algorithm.

### Example

In the following example, We can get cosine similarity from MovieLens dataset.
And then calculate predicted rating by transform method.

```scala
import com.github.takemikami.selica.ml.recommendation._

// load sample data (movielens)
case class Rating(userId: String, movieId: String, rating: Double, timestamp: Long)
def parseRating(str: String): Rating = {
  val fields = str.split("::")
  assert(fields.size == 4)
  Rating(fields(0).toString, fields(1).toString, fields(2).toDouble, fields(3).toLong)
}
val ratings = spark.read.textFile("file:///usr/local/opt/apache-spark/libexec/data/mllib/als/sample_movielens_ratings.txt").map(parseRating).toDF()
val Array(training, test) = ratings.randomSplit(Array(0.9, 0.1), seed = 12345)

// fitting
val cf = new ItemBasedCollaborativeFiltering().setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
val model = cf.fit(training)

// transform
val df = model.transform(test)
df.show()

// dump item similarity
model.similarityDataFrame.show()
```

## Item Features Similarity

Item features similarity is mainly use for recommender.
selica provide item cosine similarity computed by each item features or hashed features.

### Example

In the following example, We can get item cosine similarity from features.

```scala
import com.github.takemikami.selica.ml.recommendation._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.MinMaxScaler

// create sampledata
val itemsFeatures = Seq(
  ("hokkaido", Vectors.dense(Array(5320523, 78420.77))),
  ("tokyo", Vectors.dense(Array(13742906, 2191.0))),
  ("kyoto", Vectors.dense(Array(2599313, 4612.19))),
  ("osaka", Vectors.dense(Array(8831642, 1905.14))),
  ("okinawa", Vectors.dense(Array(1443802, 2281.14)))
).toDF("itemId", "features")

// feature scaling
val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
val scalerModel = scaler.fit(itemsFeatures)
val scaledItemsFeatures = scalerModel.transform(itemsFeatures)

// fitting
val ifs = new ItemFeatureSimilarity().setItemCol("itemId").setFeaturesCol("scaledFeatures")
val model = ifs.fit(scaledItemsFeatures)

// dump item similarity
model.similarityDataFrame.show()
```

In the following example, We can get item cosine similarity from text.

```scala
import com.github.takemikami.selica.ml.recommendation._
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

// create sampledata (from New Oxford American Dictionary)
val sentenceData = spark.createDataFrame(Seq(
  ("apple", "the round fruit of a tree of the rose family, which typically has thin red or green skin and crisp flesh. Many varieties have been developed as dessert or cooking fruit or for making cider."),
  ("strawberry", "a sweet soft red fruit with a seed-studded surface."),
  ("grape", "a berry, typically green (classified as white), purple, red, or black, growing in clusters on a grapevine, eaten as fruit, and used in making wine."),
  ("tea", "a hot drink made by infusing the dried crushed leaves of the tea plant in boiling water."),
  ("coffee", "a drink made from the roasted and ground beanlike seeds of a tropical shrub, served hot or iced")
)).toDF("item_name", "sentence")

// tokenize
val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val wordsData = tokenizer.transform(sentenceData)

// featurize
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
val featurizedData = hashingTF.transform(wordsData)

// fitting
val ifs = new ItemFeatureSimilarity().setItemCol("item_name").setFeaturesCol("rawFeatures")
val model = ifs.fit(featurizedData)

// dump item similarity
model.similarityDataFrame.show()
```
