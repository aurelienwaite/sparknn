name := "sparknn"

version := "1.0"

scalaVersion := "2.11.7"

EclipseKeys.withSource := true

// Resolvers
resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
 		    "org.apache.spark" %% "spark-core" % "1.3.0",
		    "org.apache.spark" %% "spark-mllib" % "1.3.0",
		    "org.deeplearning4j" % "dl4j-spark-ml" % "0.0.3.3.5.alpha2-SNAPSHOT",
		    "org.nd4j" % "nd4j-jblas" % "0.0.3.5.5.6-SNAPSHOT", 
		    "org.nd4j" % "nd4j-scala-api" % "0.0.3.5.5.6-SNAPSHOT", 
		    "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.4.1",
		    "org.scalatest" %% "scalatest" % "2.2.4" % "test"
		    )
		    

dependencyOverrides ++= Set(
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.4.4"
)