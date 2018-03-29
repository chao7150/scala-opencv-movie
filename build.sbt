import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.example",
      scalaVersion := "2.11.6",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "Hello",
    libraryDependencies += scalaTest % Test
  )

fork := true
javaOptions += "-Djava.library.path=.:./lib"
libraryDependencies += "org.scala-lang" % "scala-reflect" % "2.11.6"