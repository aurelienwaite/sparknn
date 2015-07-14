package com.sdl.spark.sql.sources.nplm

import java.io.File
import java.io.FileWriter

import scala.io.Source

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.SQLContext
import org.scalatest.FlatSpec
import org.scalatest.Matchers


/**
 * @author rorywaite
 */
class NPLMRelationSpec extends FlatSpec with Matchers{
  
  "The NPLM relation" should "load NPLM data" in {
    
    val tmpFile = File.createTempFile("nplm_data", ".txt")
    tmpFile.deleteOnExit()
    val writer = new FileWriter(tmpFile)
    for(c <- Source.fromInputStream(this.getClass.getResourceAsStream("/nplm_data.txt"))){
      writer.append(c)
    }
    writer.close
    
    val source = new DefaultSource
    val sparkConf = new SparkConf().setAppName("Spec").setMaster("local[1]")
    val sc = SQLContext.getOrCreate(SparkContext.getOrCreate(sparkConf))
    val relation = source.createRelation(sc, Map("path" -> tmpFile.getAbsolutePath) )
    val rows = relation.buildScan(relation.schema.fieldNames).collect()
    rows.length should be (999)
    // Check that the 8th row has been parsed correctly
    rows(7)(0) should be (0.0)
    rows(7)(1).asInstanceOf[DenseVector](2) should be (342)
  }
  
}