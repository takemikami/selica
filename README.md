selica
---

selica - Spark mllib Extend Library Implements Calculation Algorithm.

It's original library of Apache Spark MLlib, for my own use. and it's still developing.

[![Build Status](https://travis-ci.org/takemikami/selica.svg)](https://travis-ci.org/takemikami/selica)
[![Coverage Status](https://coveralls.io/repos/github/takemikami/selica/badge.svg)](https://coveralls.io/github/takemikami/selica)
[![Scaladoc](https://img.shields.io/badge/scaladoc-here-yellowgreen.svg)](http://javadoc.io/doc/com.github.takemikami/selica_2.11/)
[![Gitbook](https://img.shields.io/badge/gitbook-here-yellowgreen.svg)](https://takemikami.gitbooks.io/selica-programming-guide/content/en/)

# Overview

selica implements following algorithm.

- item-based collaborative filtering recommendation
- Japanse tokenizer by kuromoji and IPADIC

# Getting Started

## Execute example

execute spark-shell with selica.

```
$ spark-shell --repositories https://oss.sonatype.org/content/repositories/releases --packages com.github.takemikami:selica_2.11:0.0.1
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

## Build and execute

build selica.

```
$ git clone git@github.com:takemikami/selica.git
$ cd selica
$ sbt assembly
```

execute spark-shell with selica.

```
$ spark-shell --jars target/scala-2.11/selica-assembly-*-SNAPSHOT.jar
```

and then execute example.
