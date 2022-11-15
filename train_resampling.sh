#!/bin/bash


# python train_4top_resampling_lhe.py --config config_lhe_wbjet.yaml --epoch 10000 --batch 64 -o 20220110_wbjet_1 --device 2 --cla 1 --model GCN3 --fea 4 --lr 1e-3 --color 0

# python eval_4top_resampling_lhe.py --config config_lhe_wbjet.yaml --batch 1 -o 20220110_wbjet_1 --device 2 --color 0


python train_4top_resampling.py --config config_4top_re.yaml --epoch 1000 --batch 128 -o 20220622_resampling --device 0 --cla 1 --model GNN2layer_re --fea 4 --lr 1e-3

python eval_4top_resampling.py --config config_4top_re.yaml --batch 1 -o 20220622_resampling --device 0
