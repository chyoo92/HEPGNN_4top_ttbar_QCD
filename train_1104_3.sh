#!/bin/bash
# python train_4top_abs_4fea_h5.py --config config_QCD_ttbar_h5.yaml --epoch 200 --batch 1024 -o 211005_abs_QCD_ttbar_4fea --device 0 --cla 1 --model GNN1layer --fea 4

# python eval_4top_abs_4fea_h5.py --config config_QCD_ttbar_h5.yaml --batch 1024 -o 211005_abs_QCD_ttbar_4fea --device 0 --cla 1


###
python train_4top_abs_4fea_h5.py --config config_ttbar_4top_h5_all.yaml --epoch 500 --batch 1024 -o 211104_all_abs_ttbar_4top_4fea --device 0 --cla 1 --model GNN1layer --fea 4

python eval_4top_abs_4fea_h5.py --config config_ttbar_4top_h5_all.yaml --batch 1024 -o 211104_all_abs_ttbar_4top_4fea --device 0 --cla 1


# python train_4top_abs_4fea_h5.py --config config_QCD_4top_h5.yaml --epoch 200 --batch 1024 -o 211005_abs_QCD_4top_4fea --device 0 --cla 1 --model GNN1layer --fea 4

# python eval_4top_abs_4fea_h5.py --config config_QCD_4top_h5.yaml --batch 1024 -o 211005_abs_QCD_4top_4fea --device 0 --cla 1

