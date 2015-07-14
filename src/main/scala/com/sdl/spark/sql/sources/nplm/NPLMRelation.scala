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

case class NPLMRelation(location: String)(@transient val sqlContext: SQLContext) extends BaseRelation
    with PrunedScan with Logging {

  override def schema: StructType = StructType(
    StructField("label", DoubleType, nullable = false) :: 
      StructField("words", VectorUDT())
      :: Nil)

  override def buildScan(requiredColumns: Array[String]): RDD[Row] = {
    val sc = sqlContext.sparkContext
    sc.textFile(location, sc.defaultMinPartitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
        val items = line.split('\t')
        val label = items.last.toDouble
        val featureVec = Vectors.dense(items.head.split(' ').map(_.toDouble))
        Row.fromTuple((label, featureVec ))
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
    new NPLMRelation(path)(sqlContext)
  }
}
