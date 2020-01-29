# Japanese Tokenizer

Japanese tokenizer is alternative string tokenizer for Japanese.
selica provide Japanese Tokenizer by Atilika Kuromoji and IPADIC.

Atilika Kuromoji: http://www.atilika.org/

## Japanese Tokenizer by Kuromoji-IPADIC

### Example

In the following example, We can get word list from Sample Japanese sentences.

```scala
import com.github.takemikami.selica.ml.feature._

// create input data
val sentenceDataFrame = spark.createDataFrame(Seq(
  (0, "日本語の形態素解析を行います"),
  (1, "焼き肉食べたい"),
  (2, "今日も朝から眠い")
)).toDF("id", "sentence")

// initialize tokenizer
val tokenizer = new JapaneseTokenizer().setInputCol("sentence").setOutputCol("words")

// transform
val tokenized = tokenizer.transform(sentenceDataFrame)
tokenized.show()
```
