#!/bin/bash

OUTPUT="selected.csv"

rm -rf ${OUTPUT}
mvn clean package
spark-submit \
  --class com.github.benfradet.Titanic \
  --master local[2] \
  target/titanic-1.0-SNAPSHOT.jar \
  src/main/resources/train.csv src/main/resources/test.csv ${OUTPUT}

