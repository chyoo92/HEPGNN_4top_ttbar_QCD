#!/usr/bin/env python
# coding: utf-8
# %%
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch_geometric.data import InMemoryDataset as PyGDataset, Data as PyGData
from bisect import bisect_right
from glob import glob
import numpy as np
import math



   
        
class HEPGNNDataset_h5_LHE_re(PyGDataset):
    def __init__(self, **kwargs):
        super(HEPGNNDataset_h5_LHE_re, self).__init__(None, transform=None, pre_transform=None)
        self.isLoaded = False

        self.fNames = []
        self.sampleInfo = pd.DataFrame(columns=["procName", "fileName", "weight", "fileIdx"])

    def len(self):
        return int(self.maxEventsList[-1])

    def get(self, idx):
        if not self.isLoaded: self.initialize()

        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]
        idx = int(idx - offset)
     
        
        weight = self.weightList[fileIdx][idx]
        procIdxs = self.procList[fileIdx][idx]
        
        fea1 = self.f_fea1_List[fileIdx][idx]
        fea2 = self.f_fea2_List[fileIdx][idx]
        fea3 = self.f_fea3_List[fileIdx][idx]
        fea4 = self.f_fea4_List[fileIdx][idx]
        
        
        
        feats = torch.Tensor(np.concatenate((fea1.reshape(-1,1),fea2.reshape(-1,1),fea3.reshape(-1,1),fea4.reshape(-1,1)),axis=1))
  
        edge1 = self.f_edge1_List[fileIdx][idx][()]
        edge2 = self.f_edge2_List[fileIdx][idx][()]
        edges = torch.LongTensor(np.stack([edge1, edge2]))
     
        data = PyGData(x = feats, edge_index = edges)
        data.ww = torch.tensor(weight)

        return data
    def addSample(self, procName, fNamePattern, weight=1, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fNames))
        print(procName, fNamePattern)

        for fName in glob(fNamePattern):
            if not fName.endswith(".h5"): continue
            fileIdx = len(self.fNames)
            self.fNames.append(fName)

            info = {
                'procName':procName, 'weight':weight, 'nEvent':0,
                'label':0, ## default label, to be filled later
                'fileName':fName, 'fileIdx':fileIdx, 
            }
            self.sampleInfo = self.sampleInfo.append(info, ignore_index=True)




    def setProcessLabel(self, procName, label):
        self.sampleInfo.loc[self.sampleInfo.procName==procName, 'label'] = label
    def initialize(self):
        if self.isLoaded: return

        print(self.sampleInfo)
        procNames = list(self.sampleInfo['procName'].unique())




        self.weightList = []
        self.procList = []
        self.f_fea1_List = []
        self.f_fea2_List = []
        self.f_fea3_List = []
        self.f_fea4_List = []

        self.f_edge1_List = []
        self.f_edge2_List = []
        
        
        nFiles = len(self.sampleInfo)
        ## Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):
#             print("Loading files... (%d/%d) %s" % (i+1,nFiles,fName), end='\r')
            f = h5py.File(fName, 'r', libver='latest', swmr=True)
            
            nEvent = len(f['events']['m'])
        
            self.sampleInfo.loc[i, 'nEvent'] = nEvent

    

        
            f_m = f['events']['m']
            f_px = f['events']['px']
            f_py = f['events']['py']
            f_pz = f['events']['pz']
            f_id = f['events']['id']
            f_weight = f['events']['weight']
                        
            f_edge1 = f['graphs']['edge1']
            f_edge2 = f['graphs']['edge2']
                  
            self.f_fea1_List.append(f_m)
            self.f_fea2_List.append(f_px)
            self.f_fea3_List.append(f_py)
            self.f_fea4_List.append(f_pz)
              
            self.f_edge1_List.append(f_edge1)
            self.f_edge2_List.append(f_edge2)
            
            self.weightList.append(f_weight)
 
            
            procIdx = procNames.index(self.sampleInfo['procName'][i])
            self.procList.append(torch.ones(nEvent, dtype=torch.int32, requires_grad=False)*procIdx)
        
        ## Compute cumulative sums of nEvent, to be used for the file indexing
        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvent'])))

       
        print('-'*80)
        self.isLoaded = True