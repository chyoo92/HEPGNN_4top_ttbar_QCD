import torch
import os
import numpy as np
import glob
from torch_geometric.nn import radius_graph
path1 = '/store/hep/users/yewzzang/4top_QCD_ttbar/data_graph/pt/*_weight_*/*/*'
list_name1 = glob.glob(path1)

path2 = '/store/hep/users/yewzzang/4top_QCD_ttbar/data_graph/pt/*_weight_*/*'
list_name2 = glob.glob(path2)

list_name = list_name1 + list_name2

file_list_pt = [file for file in list_name if file.endswith(".pt")]

weights = np.zeros(0)
for i in range(len(file_list_pt)):
    file_name = file_list_pt[i]
    print(file_name)

    file = torch.load(file_name)
    for j in range(len(file)):
        pos = file[j].pos
        edge_index = radius_graph(pos, r=15,loop=False)
        
        file[j].edge_index = edge_index
# #         weight = file[j].x[:,6]/np.abs(file[j].x[:,6])
# #         weights = np.concatenate((weights, weight))

#         file[j].ww = file[j].x[:,6]/np.abs(file[j].x[:,6])
#         file[j].x = file[j].x[:,:6]
    test = file_name.split('/')[8:9][0].split('_')[0]
    test1 = file_name.split('/')[9:10][0]
    test2 = file_name.split('/')[-1:][0]
    if test == 'ttbar':
        print('ddd')
        save_path = '/store/hep/users/yewzzang/4top_QCD_ttbar/data_graph/pt/'+ str(test) +'_alledge/'
        save_file = save_path + str(test2)
    else:
        save_path = '/store/hep/users/yewzzang/4top_QCD_ttbar/data_graph/pt/'+ str(test) +'_alledge/' + str(test1) +'/'
        save_file = save_path + str(test2)

    if not os.path.exists(save_path): os.makedirs(save_path)
    torch.save(file, save_file)

