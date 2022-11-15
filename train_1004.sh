#!/bin/bash
# python train_4top_abs_4fea_h5.py --config config_QCD_ttbar_h5.yaml --epoch 200 --batch 1024 -o 210927_abs_QCD_ttbar_4fea --device 0 --cla 1 --model GNN1layer --fea 4

# python eval_4top_abs_4fea_h5.py --config config_QCD_ttbar_h5.yaml --batch 1024 -o 210927_abs_QCD_ttbar_4fea --device 0 --cla 1



# ###
# python train_4top_abs_4fea_h5.py --config config_ttbar_4top_h5.yaml --epoch 200 --batch 1024 -o 210927_abs_ttbar_4top_4fea --device 0 --cla 1 --model GNN1layer --fea 4

# python eval_4top_abs_4fea_h5.py --config config_ttbar_4top_h5.yaml --batch 1024 -o 210927_abs_ttbar_4top_4fea --device 0 --cla 1


python train_4top_no_abs_4fea_h5.py --config config_QCD_4top_h5.yaml --epoch 200 --batch 1024 -o 220622_test --device 0 --cla 1 --model GNN1layer --fea 4

python eval_4top_no_abs_4fea_h5.py --config config_QCD_4top_h5.yaml --batch 1024 -o 220622_test --device 0 --cla 1





# ## QCD를 tt-4top model에서 evaluation하기
# ## abs

# python eval_4top_abs_4fea_h5.py --config config_QCD_re.yaml --batch 1024 -o 211004_abs_QCD_eval_test --device 0 --cla 1

# ### -weight
# python eval_4top_no_abs_4fea_h5.py --config config_QCD_re.yaml --batch 1024 -o 211004_neg_QCD_eval_test --device 0 --cla 1

# ### abs 모델에 -weight 데이터 evaluation
# python eval_4top_no_abs_4fea_h5.py --config config_ttbar_4top_h5.yaml --batch 1024 -o 211004_neg_ttbar_4top_eval_test --device 0 --cla 1

# python eval_4top_no_abs_4fea_h5.py --config config_QCD_ttbar_h5.yaml --batch 1024 -o 211004_neg_QCD_ttbar_eval_test --device 0 --cla 1


# python eval_4top_no_abs_4fea_h5.py --config config_QCD_4top_h5.yaml --batch 1024 -o 211004_neg_QCD_4top_eval_test --device 0 --cla 1