#!/bin/bash

####### 0729 all edge eval and train

            
# python train_4top_no_abs.py --config config_QCD_ttbar_all.yaml --epoch 200 --batch 256 -o 210729_QCD_ttbar --device 0 --cla 1 --model GNN1layer

# python eval_4top_no_abs.py --config config_QCD_ttbar_all.yaml --batch 256 -o 210729_QCD_ttbar --device 0 
###

# python train_4top_no_abs.py --config config_QCD_4top_all.yaml --epoch 200 --batch 256 -o 210729_QCD_4top --device 0 --cla 1 --model GNN1layer

# python eval_4top_no_abs.py --config config_QCD_4top_all.yaml --batch 256 -o 210729_QCD_4top --device 0 

# ###
python train_4top_no_abs.py --config config_ttbar_4top_all.yaml --epoch 200 --batch 256 -o 210729_ttbar_4top --device 0 --cla 1 --model GNN1layer

python eval_4top_no_abs.py --config config_ttbar_4top_all.yaml --batch 256 -o 210729_ttbar_4top --device 0 

########## 2layer
# python train_4top_no_abs.py --config config_QCD_ttbar_all.yaml --epoch 200 --batch 256 -o 210729_QCD_ttbar_2layer --device 0 --cla 1 --model GNN2layer

# python eval_4top_no_abs.py --config config_QCD_ttbar_all.yaml --batch 256 -o 210729_QCD_ttbar_2layer --device 0

###
# python train_4top_no_abs.py --config config_QCD_4top_all.yaml --epoch 200 --batch 256 -o 210729_QCD_4top_2layer --device 0 --cla 1 --model GNN2layer

# python eval_4top_no_abs.py --config config_QCD_4top_all.yaml --batch 256 -o 210729_QCD_4top_2layer --device 0 

# ####
# python train_4top_no_abs.py --config config_ttbar_4top_all.yaml --epoch 200 --batch 256 -o 210729_ttbar_4top_2layer --device 0 --cla 1 --model GNN2layer

# python eval_4top_no_abs.py --config config_ttbar_4top_all.yaml --batch 256 -o 210729_ttbar_4top_2layer --device 0 

