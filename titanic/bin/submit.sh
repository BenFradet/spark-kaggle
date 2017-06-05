#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
OUTPUT="classified.csv"
TMP_FILE="${OUTPUT}2"

sbt "project titanic" clean assembly

cd ${DIR}
rm -rf ${OUTPUT}
rm -rf ${TMP_FILE}

spark-submit \
  --class io.github.benfradet.Titanic \
  --master local[2] \
  target/scala-2.11/titanic-assembly-1.0.jar \
  src/main/resources/train.csv src/main/resources/test.csv ${OUTPUT}

mv ${OUTPUT}/part-* ${TMP_FILE}
rm -rf ${OUTPUT}
mv ${TMP_FILE} ${OUTPUT}
