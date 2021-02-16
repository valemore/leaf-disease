#!/bin/sh
mkdir -p submission/leaf
mkdir -p submission/folds
cp leaf/dta.py submission/leaf/
cp b4-folds/b4-446_fold0.Feb14_20-26-07/b4-446_fold0.Feb14_20-26-07-10 submission/folds/1.pth
cp b4-folds/b4-446_fold1.Feb14_22-09-33/b4-446_fold1.Feb14_22-09-33-11 submission/folds/2.pth
cp b4-folds/b4-446_fold2.Feb14_23-51-32/b4-446_fold2.Feb14_23-51-32-8 submission/folds/3.pth
cp b4-folds/b4-446_fold3.Feb15_01-32-45/b4-446_fold3.Feb15_01-32-45-11 submission/folds/4.pth
cp b4-folds/b4-446_fold4.Feb15_03-15-34/b4-446_fold4.Feb15_03-15-34-9 submission/folds/5.pth
cd submission
rm -f leaf.zip
rm -f folds.zip
zip -r leaf.zip leaf
cd folds
zip -r ../folds.zip ./*
