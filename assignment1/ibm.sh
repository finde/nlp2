#!/usr/bin/env bash
#align.sh [ibmModel] [englishData] [frenchData]
ibmModel=$1
sourceData=$2
targetData=$3

if [ $ibmModel==1 ]
then
    python ibmmodel/model1.py $sourceData $targetData data/test.e data/test.f
fi

if [ $ibmModel==2 ]
then
    python ibmmodel/model1.py $sourceData $targetData data/test.e data/test.f
fi

if [ $ibmModel==Moore ]
then
    python ibmmodel/model1_moore.py $sourceData $targetData data/test.e data/test.f
fi

