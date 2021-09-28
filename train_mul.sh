#!/bin/bash

# python train_4top_no_abs_4fea_h5.py --config config_all_h5.yaml --epoch 200 --batch 2048 -o 210917_all_4fea_1layer --device 0 --cla 3 --model GNN1layer_mul --fea 4

# python eval_4top_no_abs_4fea_h5.py --config config_all_h5.yaml --batch 2048 -o 210917_all_4fea_1layer --device 0


# python train_4top_no_abs_4fea_h5.py --config config_all_h5.yaml --epoch 200 --batch 2048 -o 210917_all_4fea_2layer --device 0 --cla 3 --model GNN2layer_mul --fea 4

# python eval_4top_no_abs_4fea_h5.py --config config_all_h5.yaml --batch 2048 -o 210917_all_4fea_2layer --device 0




# python train_4top_abs_4fea_h5.py --config config_all_h5.yaml --epoch 200 --batch 2048 -o 210917_all_4fea_abs_1layer --device 0 --cla 3 --model GNN1layer_mul --fea 4

# python eval_4top_abs_4fea_h5.py --config config_all_h5.yaml --batch 2048 -o 210917_all_4fea_abs_1layer --device 0


# python train_4top_abs_4fea_h5.py --config config_all_h5.yaml --epoch 200 --batch 2048 -o 210917_all_4fea_abs_2layer --device 0 --cla 3 --model GNN2layer_mul --fea 4

# python eval_4top_abs_4fea_h5.py --config config_all_h5.yaml --batch 2048 -o 210917_all_4fea_abs_2layer --device 0

python eval_4top_no_abs_4fea_h5.py --config config_all_h5.yaml --batch 2048 -o 210914_all_4fea_1layer --device 0 --cla 3

python eval_4top_no_abs_4fea_h5.py --config config_all_h5.yaml --batch 2048 -o 210914_all_4fea_2layer --device 0 --cla 3
