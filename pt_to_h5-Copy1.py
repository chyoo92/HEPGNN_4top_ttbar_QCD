#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import numpy as np
import glob
import h5py
# from torch_geometric.nn import radius_graph
path1 = '/store/hep/users/yewzzang/4top_QCD_ttbar/data_graph/pt/*_alledge/*/*'
list_name1 = glob.glob(path1)

path2 = '/store/hep/users/yewzzang/4top_QCD_ttbar/data_graph/pt/*_alledge/*'
list_name2 = glob.glob(path2)

list_name = list_name1 + list_name2

file_list_pt = [file for file in list_name if file.endswith(".pt")]


# In[2]:


for i in range(len(file_list_pt)):
    file_name = file_list_pt[i]
    print(file_name)
    
    edge_index_p = []
    pos_p = []
    x_p = []
    y_p = []
    
    file = torch.load(file_name)
    for j in range(len(file)):
        
        
        edge_index_p.append(file[j].edge_index)
        pos_p.append(file[j].pos)
        x_p.append(file[j].x)
        y_p.append(file[j].y)

    edge_flat = [[] for _ in range(len(y_p))]
    pos_flat = [[] for _ in range(len(y_p))]
    x_flat = [[] for _ in range(len(y_p))]
#     y_flat = [[] for _ in range(len(y_p))]
    y_flat = y_p
    for k in range(len(y_p)):
        
        edge_f = [y for x in edge_index_p[k] for y in x]
       
        pos_f = [y for x in pos_p[k] for y in x]
        x_f = [y for x in x_p[k] for y in x]

        
        edge_flat[k] = np.asarray(edge_f)

        pos_flat[k] = np.asarray(pos_f)
        x_flat[k] = np.asarray(x_f)

    test = file_name.split('/')[8:9][0].split('_')[0]
    test1 = file_name.split('/')[9:10][0]
    test2 = file_name.split('/')[-1:][0].split('.')[0]

    if test == 'ttbar':
    
        save_path = '/store/hep/users/yewzzang/4top_QCD_ttbar/data_graph/pt/'+ str(test) +'_alledge_h5/'
        save_file = save_path + str(test2)
    else:
        save_path = '/store/hep/users/yewzzang/4top_QCD_ttbar/data_graph/pt/'+ str(test) +'_alledge_h5/' + str(test1) +'/'
        save_file = save_path + str(test2)

    if not os.path.exists(save_path): os.makedirs(save_path)



    f = h5py.File(save_file +'.h5', mode='w')

    dt = h5py.special_dtype(vlen=np.dtype('float32'))
    g = f.create_group("group")
    
    pos = g.create_group("pos")
    x_fea = g.create_group("fea")
    edge_index = g.create_group("edge")
    label = g.create_group("label")
    
    pos.create_dataset('pos', (len(pos_flat),), dtype=dt)
    pos['pos'][...] = pos_flat
    x_fea.create_dataset('fea', (len(pos_flat),), dtype=dt)
    x_fea['fea'][...] = x_flat
    edge_index.create_dataset('edge', (len(pos_flat),), dtype=dt)
    edge_index['edge'][...] = edge_flat
    label.create_dataset('label', (len(pos_flat),), dtype=dt)
    label['label'][...] = y_flat
 
    f.close()


