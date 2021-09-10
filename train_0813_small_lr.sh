#!/bin/bash
python train_4top_no_abs_4fea_h5.py --config config_QCD_ttbar_h5.yaml --epoch 200 --batch 1024 -o 210813_QCD_ttbar_4fea_sm_lr --device 0 --cla 1 --model GNN1layer --fea 4 --lr 1e-6

python eval_4top_no_abs_4fea_h5.py --config config_QCD_ttbar_h5.yaml --batch 1024 -o 210813_QCD_ttbar_4fea_sm_lr --device 0 


python train_4top_no_abs_4fea_h5.py --config config_QCD_ttbar_h5.yaml --epoch 200 --batch 1024 -o 210813_QCD_ttbar_4fea_2l_sm_lr --device 0 --cla 1 --model GNN2layer --fea 4 --lr 1e-6

python eval_4top_no_abs_4fea_h5.py --config config_QCD_ttbar_h5.yaml --batch 1024 -o 210813_QCD_ttbar_4fea_2l_sm_lr --device 0 

# ###
# python train_4top_no_abs_4fea_h5.py --config config_ttbar_4top_h5.yaml --epoch 200 --batch 1024 -o 210813_ttbar_4top_4fea_sm_lr --device 0 --cla 1 --model GNN1layer --fea 4 --lr 1e-6

# python eval_4top_no_abs_4fea_h5.py --config config_ttbar_4top_h5.yaml --batch 1024 -o 210813_ttbar_4top_4fea_sm_lr --device 0 


# python train_4top_no_abs_4fea_h5.py --config config_QCD_4top_h5.yaml --epoch 200 --batch 1024 -o 210813_QCD_4top_4fea_2l_sm_lr --device 0 --cla 1 --model GNN2layer --fea 4 --lr 1e-6

# python eval_4top_no_abs_4fea_h5.py --config config_QCD_4top_h5.yaml --batch 1024 -o 210813_QCD_4top_4fea_2l_sm_lr --device 0 


# python train_4top_no_abs_4fea_h5.py --config config_QCD_4top_h5.yaml --epoch 200 --batch 1024 -o 210813_QCD_4top_4fea_2l_sm_lr --device 0 --cla 1 --model GNN2layer --fea 4 --lr 1e-6

# python eval_4top_no_abs_4fea_h5.py --config config_QCD_4top_h5.yaml --batch 1024 -o 210813_QCD_4top_4fea_2l_sm_lr --device 0 

# ###
# python train_4top_no_abs_4fea_h5.py --config config_ttbar_4top_h5.yaml --epoch 200 --batch 1024 -o 210813_ttbar_4top_4fea_2l_sm_lr --device 0 --cla 1 --model GNN2layer --fea 4 --lr 1e-6

# python eval_4top_no_abs_4fea_h5.py --config config_ttbar_4top_h5.yaml --batch 1024 -o 210813_ttbar_4top_4fea_2l_sm_lr --device 0 
