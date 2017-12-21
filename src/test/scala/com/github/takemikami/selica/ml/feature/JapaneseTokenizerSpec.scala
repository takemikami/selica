package com.github.takemikami.selica.ml.feature

import org.scalatest._
import Matchers._
import com.github.takemikami.selica.SparkSessionForUnitTest
import org.apache.spark.sql.SparkSession

class JapaneseTokenizerSpec extends FlatSpec with BeforeAndAfter  {

  private var sparkSession: SparkSession = _
  before {
    sparkSession = SparkSessionForUnitTest.getSession()
  }
  after {
  }

  "ItemBasedCollaborativeFiltering" should "can fit & transform" in {
    val spark = sparkSession
    import spark.implicits._

    val sentences = Seq(
      ("0", "日本語の形態素解析を行います"),
      ("1", "焼き肉食べたい"),
      ("2", "今日も朝から眠い")
    ).toDF("textId", "sentence")

    val tokenizer = new JapaneseTokenizer().setInputCol("sentence").setOutputCol("words")

    // transform
    val df = tokenizer.transform(sentences)

    df.count() shouldEqual 3
    df.columns.contains("sentence") shouldEqual true
    df.columns.contains("words") shouldEqual true
  }

  "JapaneseTokenizer" should "can tokenize by kuromoji" in {
    val words = JapaneseTokenizer.tokenize("日本語の形態素解析を行います")
    words.length > 0 shouldEqual true
    words(0).startsWith("日") shouldEqual true
    words(words.length-1).endsWith("す") shouldEqual true
  }

}
