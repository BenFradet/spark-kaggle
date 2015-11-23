#!/bin/bash

OUTPUT="src/main/resources/classified.csv"
TMP_FILE="${OUTPUT}2"
NB_THREADS=2

rm -rf ${OUTPUT}
rm -rf ${TMP_FILE}

mvn clean package
spark-submit \
  --class com.github.benfradet.Titanic \
  --master local[${NB_THREADS}] \
  target/titanic-1.0-SNAPSHOT.jar \
  src/main/resources/train.csv src/main/resources/test.csv ${OUTPUT}

touch ${TMP_FILE}
for (( i=0; i<${NB_THREADS}; i++ )); do
    PART_FILE="${OUTPUT}/part-0000${i}"
    if [ ${i} == 0 ]; then
        cat ${PART_FILE} >> ${TMP_FILE}
    else
        tail -n +2 ${PART_FILE} >> ${TMP_FILE}
    fi
done

rm -rf ${OUTPUT}
mv ${TMP_FILE} ${OUTPUT}

