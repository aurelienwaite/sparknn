package com.sdl.spark.sql.sources.nplm

import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.sources.BaseRelation
import org.apache.spark.sql.sources.PrunedScan
import org.apache.spark.Logging
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.DoubleType
import org.deeplearning4j.spark.sql.types.VectorUDT
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{ SparseVector, DenseVector, Vector, Vectors }
import org.apache.spark.sql.sources.RelationProvider

/**
 * @author rorywaite
 */

case class NPLMRelation(location: String, order : Int, vocabSize : Int)(@transient val sqlContext: SQLContext) extends BaseRelation
    with PrunedScan with Logging {

  override def schema: StructType = StructType(
    StructField("label", DoubleType, nullable = false) :: 
      (1 until order).map(order => StructField("features_"+order, VectorUDT())).toList 
      ::: Nil)

  override def buildScan(requiredColumns: Array[String]): RDD[Row] = {
    val sc = sqlContext.sparkContext
    sc.textFile(location, sc.defaultMinPartitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
        val items = line.split('\t')
        val label = items.last.toDouble
        val featureVecs = items.head.split(' ').map(_.toInt)
        .map(i => Vectors.sparse(vocabSize, List((i, 1.0))))
        Row.fromSeq((label :: featureVecs.toList ))
      }
  }

  override def hashCode(): Int = 41 * (41 + location.hashCode) + schema.hashCode()

  override def equals(other: Any): Boolean = other match {
    case that: NPLMRelation =>
      (this.location == that.location) && this.schema.equals(that.schema)
    case _ => false
  }

}

class DefaultSource extends RelationProvider {

  override def createRelation(sqlContext: SQLContext, parameters: Map[String, String]) = {
    val path = parameters.getOrElse("path", sys.error("'path' must be specified for NPLM data."))
    val order = parameters.getOrElse("order", sys.error("'order' must be specified for NPLM data."))
    val vocabSize = parameters.getOrElse("vocabSize", sys.error("'vocabSize' must be specified for NPLM data."))
    new NPLMRelation(path, order.toInt, vocabSize.toInt)(sqlContext)
  }
}
