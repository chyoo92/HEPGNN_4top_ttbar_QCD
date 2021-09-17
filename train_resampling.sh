#!/bin/bash



python train_4top_resampling.py --config config_QCD_re.yaml --epoch 200 --batch 2048 -o 210917_QCD_1layer_re --device 0 --cla 1 --model GNN1layer_re --fea 4

python eval_4top_resampling5.py --config config_QCD_re.yaml --batch 2048 -o 210917_QCD_1layer_re --device 0


python train_4top_resampling.py --config config_QCD_re.yaml --epoch 200 --batch 2048 -o 210917_QCD_2layer_re --device 0 --cla 1 --model GNN2layer_re --fea 4

python eval_4top_resampling.py --config config_QCD_re5.yaml --batch 2048 -o 210917_QCD_2layer_re --device 0


python train_4top_resampling.py --config config_ttbar_re.yaml --epoch 200 --batch 2048 -o 210917_ttbar_1layer_re --device 0 --cla 1 --model GNN1layer_re --fea 4

python eval_4top_resampling5.py --config config_ttbar_re.yaml --batch 2048 -o 210917_ttbar_1layer_re --device 0


python train_4top_resampling.py --config config_ttbar_re.yaml --epoch 200 --batch 2048 -o 210917_ttbar_2layer_re --device 0 --cla 1 --model GNN2layer_re --fea 4

python eval_4top_resampling.py --config config_ttbar_re5.yaml --batch 2048 -o 210917_ttbar_2layer_re --device 0


python train_4top_resampling.py --config config_4top_re.yaml --epoch 200 --batch 2048 -o 210917_4top4top_1layer_re --device 0 --cla 1 --model GNN1layer_re --fea 4

python eval_4top_resampling5.py --config config_4top_re.yaml --batch 2048 -o 210917_4top_1layer_re --device 0


python train_4top_resampling.py --config config_4top_re.yaml --epoch 200 --batch 2048 -o 210917_4top_2layer_re --device 0 --cla 1 --model GNN2layer_re --fea 4

python eval_4top_resampling.py --config config_4top_re5.yaml --batch 2048 -o 210917_4top_2layer_re --device 0


