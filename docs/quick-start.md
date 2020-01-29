# Quick Start

Building selica and executing spark-shell with selica take few minutes.

## Prerequisites

selica require following software.

- sbt
- Apache Spark 2.2.0, spark-shell ready

## Execute selica

Execute spark-shell with selica.

```
$ spark-shell --repositories https://oss.sonatype.org/content/repositories/releases --packages com.github.takemikami:selica_2.11:0.0.2
```

## Build and Execute selica

You can build and execute from source code, if you need.

### Building selica

Cloning GitHub selica repository:

```
$ git clone git@github.com:takemikami/selica.git
```

selica can build by sbt:

```
$ cd selica
$ sbt assembly
```

You can get ``selica-assembly-*-SNAPSHOT.jar`` under ``target/scala-2.11``.

### Executing spark-shell with selica

Executing spark-shell with selica package.

```
$ spark-shell --jars target/scala-2.11/selica-assembly-*-SNAPSHOT.jar
```
