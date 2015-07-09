name := "sparknn"

version := "1.0"

scalaVersion := "2.10.4"

// Resolvers
resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
 		    "org.apache.spark" %% "spark-core" % "1.4.0",
		    "org.apache.spark" %% "spark-mllib" % "1.4.0",
		    "org.deeplearning4j" % "dl4j-spark-ml" % "0.0.3.3.5.alpha2-SNAPSHOT",
		    "org.nd4j" % "nd4j-scala-api" % "0.0.3.5.5.6-SNAPSHOT", 
		    "org.scalatest" % "scalatest_2.10" % "2.2.4" % "test"
		    )
		    

dependencyOverrides ++= Set(
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.4.4"
)