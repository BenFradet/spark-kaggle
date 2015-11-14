#!/bin/bash

rm -rf selected.csv
mvn clean package
spark-submit \
  --class com.github.benfradet.Titanic \
  --master local[2] \
  target/titanic-1.0-SNAPSHOT.jar \
  src/main/resources/train.csv
