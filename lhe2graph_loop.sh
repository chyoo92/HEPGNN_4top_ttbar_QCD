#!/bin/bash


for i in {8,9,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98}
do

    python lhe2graph-Copy1.py -i /users/yewzzang/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/4top_dump_ex1/220324_055935/0000/4top_ex1_${i}.lhe -o /users/yewzzang/TTTT_TuneCP5_13TeV-amcatnlo-pythia8/4top_dump_ex1/220324_055935/graph/4top_23/4top_ex1_${i}.h5 -v
done