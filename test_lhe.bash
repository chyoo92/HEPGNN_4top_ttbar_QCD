#!/bin/bash
while read line
do
	
	echo $line
	python lhe2graph.py -i $line -o /store/hep/users/yewzzang/4top_lhe/wjet_lhe/ -v

done < /store/hep/users/yewzzang/4top_lhe/files__WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8__RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1__MINIAODSIM.txt
