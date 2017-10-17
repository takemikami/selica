selica
---

selica - Spark mllib Extend Library Implements Calculation Algorithm.

It's original library of Apache Spark MLlib, for my own use. and it's still developing.

[![Build Status](https://travis-ci.org/takemikami/selica.svg?branch=cf)](https://travis-ci.org/takemikami/selica)

# Overview

selica implements following algorithm.

- item-based collaborative filtering recommendation


# Getting Started

build selica.

```
$ git clone git@github.com:takemikami/selica.git
$ cd selica
$ sbt package
```

execute spark-shell with selica.

```
$ spark-shell --jars target/scala-2.11/selica_2.11-0.0.1-SNAPSHOT.jar
```

execute sample.

```
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
val cf = new com.github.takemikami.selica.ml.recommendation.ItemBasedCollaborativeFiltering().setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
val model = cf.fit(training)

// transform
val df = model.transform(test)
df.show()

// dump item similarity
model.similarityDataFrame.show()
```
