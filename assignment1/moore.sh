#!/usr/bin/env bash
corps=25000
echo "BASELINE"
python ibmmodel/model1_moore.py data/hansards.36.2.e data/hansards.36.2.f data/test.e data/test.f uniform_baseline $corps 0 0 1
echo "n-smoothing..."
python ibmmodel/model1_moore.py data/hansards.36.2.e data/hansards.36.2.f data/test.e data/test.f uniform_n_0_01 $corps 0.01 0 1
python ibmmodel/model1_moore.py data/hansards.36.2.e data/hansards.36.2.f data/test.e data/test.f uniform_n_0_05 $corps 0.05 0 1
python ibmmodel/model1_moore.py data/hansards.36.2.e data/hansards.36.2.f data/test.e data/test.f uniform_n_0_10 $corps 0.10 0 1
python ibmmodel/model1_moore.py data/hansards.36.2.e data/hansards.36.2.f data/test.e data/test.f uniform_n_0_15 $corps 0.15 0 1

python ibmmodel/model1_moore.py data/hansards.36.2.e data/hansards.36.2.f data/test.e data/test.f uniform_n_0__10000__1 25000 0 10000 1
python ibmmodel/model1_moore.py data/hansards.36.2.e data/hansards.36.2.f data/test.e data/test.f uniform_n_0__50000__1 25000 0 50000 1
python ibmmodel/model1_moore.py data/hansards.36.2.e data/hansards.36.2.f data/test.e data/test.f uniform_n_0__100000__1 25000 0 100000 1

python ibmmodel/model1_moore.py data/hansards.36.2.e data/hansards.36.2.f data/test.e data/test.f uniform_n_0__0__2 25000 0 0 2
python ibmmodel/model1_moore.py data/hansards.36.2.e data/hansards.36.2.f data/test.e data/test.f uniform_n_0__0__3 25000 0 0 3

python ibmmodel/model1_moore.py data/hansards.36.2.e data/hansards.36.2.f data/test.e data/test.f uniform_n_0_01__100000__3 25000 0.01 100000 3