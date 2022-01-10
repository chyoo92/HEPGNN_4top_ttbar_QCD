#!/bin/bash

# python train_4top_abs_4fea_h5.py --config config_test_h5.yaml --epoch 500 --batch 8 -o 211104_grav --device 0 --cla 1 --model GravNet --fea 4

python train_4top_abs_4fea_h5.py --config config_test_h5.yaml --epoch 500 --batch 1024 -o testttt --device 0 --cla 1 --model GNN1layer --fea 4

# python eval_4top_abs_4fea_h5.py --config config_QCD_4top_h5_all.yaml --batch 1024 -o 211104_grav --device 0 --cla 1



# python eval_4top_abs_4fea_h5.py --config config_QCD_4top_h5_all.yaml --batch 1024 -o hoho --device 0 --cla 1

# python eval_4top_abs_4fea_h5.py --config config_QCD_ttbar_h5_all.yaml --batch 1024 -o QCD_ttbar_imsi --device 0 --cla 1