/*
 * Copyright (C) 2017 Takeshi Mikami.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.takemikami.selica.ml.feature

import com.atilika.kuromoji.ipadic.Token
import com.atilika.kuromoji.ipadic.Tokenizer
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}

class JapaneseTokenizer (override val uid: String)
  extends UnaryTransformer[String, Seq[String], JapaneseTokenizer] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("selicaJpTok"))

  override protected def createTransformFunc: String => Seq[String] = {
    JapaneseTokenizer.tokenize(_)
  }

  override protected def outputDataType: DataType = new ArrayType(StringType, true)
}

object JapaneseTokenizer {
  val tokenizer = (new Tokenizer.Builder()).build()
  def tokenize(sentence: String): Array[String] = {
    val tokens = tokenizer.tokenize(sentence).toArray
    val words = for (t <- tokens; token = t.asInstanceOf[Token]) yield token.getSurface
    return words
  }
}
