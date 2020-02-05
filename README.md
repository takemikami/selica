selica
---

[![CircleCI](https://circleci.com/gh/takemikami/selica.svg?style=svg)](https://circleci.com/gh/takemikami/selica)
[![Coverage Status](https://coveralls.io/repos/github/takemikami/selica/badge.svg)](https://coveralls.io/github/takemikami/selica)
[![Scaladoc](https://img.shields.io/badge/scaladoc-here-yellowgreen.svg)](http://javadoc.io/doc/com.github.takemikami/selica_2.11/)
[![Document](https://img.shields.io/badge/document-here-yellowgreen.svg)](https://takemikami.github.io/selica/)
[![Maven Central](https://img.shields.io/maven-central/v/com.github.takemikami/selica_2.11.svg)](https://search.maven.org/#search%7Cga%7C1%7Cg%3A%22com.github.takemikami%22%20AND%20a%3A%22selica_2.11%22)

selica - Spark mllib Extend Library Implements Calculation Algorithm.

It's original library of Apache Spark MLlib, for my own use. and it's still developing.

# Overview

selica implements following algorithm.

- Item-based collaborative filtering recommendation
- Frequent pattern mining
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

### Scala

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

### Python

create python environment.

```
$ cd python
$ python -m venv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```

set SPARK_HOME environment variable, and execute unit tests.

```
$ pytest tests/
```
