#!/bin/bash
# python train_4top_no_abs_4fea_h5.py --config config_test_h5.yaml --epoch 2 --batch 64 -o 210909_all_4fea --device 3 --cla 3 --model GNN1layer --fea 4
# python eval_4top_no_abs_4fea_h5.py --config config_test_h5.yaml --batch 64 -o 210909_all_4fea --device 3 --cla 3



python train_4top_no_abs_4fea_h5.py --config config_all_h5.yaml --epoch 200 --batch 2048 -o 210914_all_4fea_1layer --device 0 --cla 3 --model GNN1layer --fea 4

python eval_4top_no_abs_4fea_h5.py --config config_all_h5.yaml --batch 2048 -o 210914_all_4fea_1layer --device 0


# python train_4top_no_abs_4fea_h5.py --config config_all_h5.yaml --epoch 200 --batch 2048 -o 210909_all_4fea_2layer --device 3 --cla 3 --model GNN2layer --fea 4

# python eval_4top_no_abs_4fea_h5.py --config config_all_h5.yaml --batch 2048 -o 210909_all_4fea_2layer --device 3 

