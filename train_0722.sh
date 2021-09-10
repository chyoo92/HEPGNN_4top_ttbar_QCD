#!/bin/bash
            
# python train_4top_0713.py --config config_QCD_ttbar.yaml --epoch 200 --batch 128 -o 210803_QCD_ttbar --device 0 --cla 1 --model GNN1layer

# python eval_4top_0713.py --config config_QCD_ttbar.yaml --batch 128 -o 210803_QCD_ttbar --device 0 



# python train_4top_0713.py --config config_QCD_4top.yaml --epoch 200 --batch 128 -o 210803_QCD_4top --device 0 --cla 1 --model GNN1layer

# python eval_4top_0713.py --config config_QCD_4top.yaml --batch 128 -o 210803_QCD_4top --device 0 





# python train_4top_0713.py --config config_ttbar_4top.yaml --epoch 200 --batch 128 -o 210803_ttbar_4top --device 0 --cla 1 --model GNN1layer

# python eval_4top_0713.py --config config_ttbar_4top.yaml --batch 128 -o 210803_ttbar_4top --device 0 


          
            
# python train_4top_0713.py --config config_QCD_ttbar.yaml --epoch 200 --batch 128 -o 210803_QCD_ttbar_2layer --device 0 --cla 1 --model GNN2layer

# python eval_4top_0713.py --config config_QCD_ttbar.yaml --batch 128 -o 210803_QCD_ttbar_2layer --device 0 



# python train_4top_0713.py --config config_QCD_4top.yaml --epoch 200 --batch 128 -o 210803_QCD_4top_2layer --device 0 --cla 1 --model GNN2layer

# python eval_4top_0713.py --config config_QCD_4top.yaml --batch 128 -o 210803_QCD_4top_2layer --device 0 



python train_4top_0713.py --config config_ttbar_4top.yaml --epoch 200 --batch 128 -o 210803_ttbar_4top_2layer --device 0 --cla 1 --model GNN2layer

python eval_4top_0713.py --config config_ttbar_4top.yaml --batch 128 -o 210803_ttbar_4top_2layer --device 0 

