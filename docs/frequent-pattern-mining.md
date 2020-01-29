# Frequent Pattern Mining

Frequent Pattern Mining algorithms for finding frequent user behavior pattern. 

## View and Conversion Pattern Mining

View and Conversion Pattern Mining for recommender. selica provide frequent conversion after view pattern finder.

### Example

In the following example, You can get conversions prediction from view pattern.

```scala
import com.github.takemikami.selica.ml.fpm._

// create sample data
val dataset = spark.createDataset(Seq(
  (1, Array(1,2,3), Array(1)),
  (1, Array(4,5,6), Array(4,6)),
  (2, Array(2,3,4), Array(1))
)).toDF("userId", "antecedent", "consequent")

// fitting
val fpm = new FrequentViewConversionPatternMining().setAntecedentCol("antecedent").setConsequentCol("consequent")
val model = fpm.fit(dataset)

// prediction
val predict = model.transform(dataset)
predict.show()

// dump fpm model
model.ruleDataFrame.show()
```

In the following example, You can get prediction model from time-series view and conversion logs.

```scala
import com.github.takemikami.selica.ml.fpm._
import org.apache.spark.sql.Row

// create sample data
val dataset = spark.createDataset(Seq(
  (1, 1, 0, 1),
  (1, 2, 0, 2),
  (1, 3, 0, 3),
  (1, 1, 1, 4),
  (1, 4, 0, 5),
  (1, 5, 0, 6),
  (1, 6, 0, 7),
  (1, 4, 1, 8),
  (1, 6, 1, 9),
  (2, 2, 0, 10),
  (2, 3, 0, 11),
  (2, 4, 0, 12),
  (2, 1, 1, 13)
)).toDF("userId", "itemId", "cv", "ts")

// transform for FrequentViewConversionPatternMining trainer
def splitTransaction(rowList: List[Row]): List[List[Row]]  = {
  val (i, j) = rowList.span((k: Row) => k(2) == 1)
  val (a, b) = j.span((k: Row) => k(2) != 1)
  b.isEmpty match {
      case true  => List[List[Row]](i ++ a)
      case false => List[List[Row]](i ++ a) ++ splitTransaction(b)
  }
}
def splitViewConversion(tran: List[Row]): (List[Int], List[Int]) = {
  val (i, j) = tran.span((k: Row) => k(2) != 1)
  (i.map(k => k(1)).toList.filter(k => k.isInstanceOf[Int]).map(k => k.asInstanceOf[Int]),
   j.map(k => k(1)).toList.filter(k => k.isInstanceOf[Int]).map(k => k.asInstanceOf[Int]))
}
val df = dataset.rdd.map { row => row(0) -> Seq(row) }.reduceByKey((k, v) => k ++ v).map{ v =>
  splitTransaction(v._2.toList.reverse).map ( u => splitViewConversion(u.reverse) )
}.flatMap(k => k).toDF("antecedent", "consequent")
df.show()

// fitting
val fpm = new FrequentViewConversionPatternMining().setAntecedentCol("antecedent").setConsequentCol("consequent")
val model = fpm.fit(df)

// dump fpm model
model.ruleDataFrame.show()
```
