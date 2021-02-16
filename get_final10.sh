#!/bin/sh
mkdir -p submission/final10
gsutil cp gs://dlspace/cassava/final10/outputs/b4-446-final10_fold0.Feb17_02-35-34/b4-446-final10_fold0.Feb17_02-35-34-11 ./submission/final10/final10_fold0.pth
gsutil cp gs://dlspace/cassava/final10/outputs/b4-446-final10_fold1.Feb17_03-58-10/b4-446-final10_fold1.Feb17_03-58-10-11 ./submission/final10/final10_fold1.pth
gsutil cp gs://dlspace/cassava/final10/outputs/b4-446-final10_fold2.Feb17_05-18-49/b4-446-final10_fold2.Feb17_05-18-49-11 ./submission/final10/final10_fold2.pth
gsutil cp gs://dlspace/cassava/final10/outputs/b4-446-final10_fold3.Feb17_06-40-27/b4-446-final10_fold3.Feb17_06-40-27-11 ./submission/final10/final10_fold3.pth
gsutil cp gs://dlspace/cassava/final10/outputs/b4-446-final10_fold4.Feb17_08-03-11/b4-446-final10_fold4.Feb17_08-03-11-11 ./submission/final10/final10_fold4.pth
