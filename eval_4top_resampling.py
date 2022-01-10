#!/usr/bin/env python
import numpy as np
import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import DataLoader
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Data
import sys, os
import subprocess
import csv, yaml
import math
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.optim as optim
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri

sys.path.append("./python")

parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output file')
#parser.add_argument('-t', '--train', action='store', type=str, required=True, help='Path to training results directory')
parser.add_argument('-a', '--all', action='store_true', help='use all events for the evaluation, no split')
parser.add_argument('--cla', action='store', type=int, default=3, help='# class')

parser.add_argument('--device', action='store', type=int, default=0, help='device name')
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('--seed', action='store', type=int, default=12345, help='random seed')
args = parser.parse_args()

config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
if args.seed: config['training']['randomSeed1'] = args.seed

sys.path.append("./python")

torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)

##### Define dataset instance #####
from dataset.HEPGNNDataset_h5_fea4_re import *
dset = HEPGNNDataset_h5_fea4_re()

for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(name, sampleInfo['path'], weight=sampleInfo['xsec']/sampleInfo['ngen'])
    dset.setProcessLabel(name, sampleInfo['label'])
dset.initialize()
lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed1'])
kwargs = {'num_workers':min(config['training']['nDataLoaders'], os.cpu_count()),
          'batch_size':args.batch, 'pin_memory':False}

if args.all:
    testLoader = DataLoader(dset, **kwargs)
else:
    trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)
    #testLoader = DataLoader(trnDset, **kwargs)
    #testLoader = DataLoader(valDset, **kwargs)
    testLoader = DataLoader(testDset, **kwargs)
torch.manual_seed(torch.initial_seed())

##### Define model instance #####
from model.allModel import *

model = torch.load('result/' + args.output+'/model.pth', map_location='cpu')
model.load_state_dict(torch.load('result/' + args.output+'/weight.pth', map_location='cpu'))


device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'

dd = 'result/' + args.output + '/train.csv'

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 5
plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.loc"] = 'upper left'
plt.rcParams['figure.figsize'] = (4*2, 3.5*3)

ax1 = plt.subplot(3, 2, 1, yscale='log', ylabel='Loss(train)', xlabel='epoch')
ax2 = plt.subplot(3, 2, 2, yscale='log', ylabel='Loss(val)', xlabel='epoch')
ax3 = plt.subplot(3, 2, 3, ylabel='Accuracy(train)', xlabel='epoch')
ax4 = plt.subplot(3, 2, 4, ylabel='Accuracy(val)', xlabel='epoch')
#ax1.set_ylim([3e-2,1e-0])
#ax2.set_ylim([3e-2,1e-0])
ax3.set_ylim([0.50,1])
ax4.set_ylim([0.50,1])
for ax in (ax1, ax2, ax3, ax4):
    ax.grid(which='major', axis='both', linestyle='-.')
    ax.grid(which='minor', linestyle=':')
    ax.set_xlim([0,400])
lines, labels = [], []

dff = pd.read_csv(dd)

label = dd.split('/')[-1].replace('__', ' ').replace('_', '=')

l = ax1.plot(dff['loss'], '.-', label=label)
ax2.plot(dff['val_loss'], '.-', label=label)

ax3.plot(dff['acc'], '.-', label=label)
ax4.plot(dff['val_acc'], '.-', label=label)

lines.append(l[0])
labels.append(label)

ax5 = plt.subplot(3,1,3)
ax5.legend(lines, labels)
ax5.axis('off')

plt.tight_layout()
plt.savefig('result/' + args.output + '/' + args.output + '_acc_loss.png', dpi=300)

#plt.show()
#plt.close()
plt.clf()



#### Start evaluation #####
from tqdm import tqdm
labels, preds = [], []
weights = []
scaledWeights = []

poss = []
batch_size = []
features = []
btags = []

procIdxs = []
fileIdxs = []
idxs = []
model.eval()
val_loss, val_acc = 0., 0.
# for i, (data, label0, weight, rescale, procIdx, fileIdx, idx, dT, dVertex, vertexX, vertexY, vertexZ) in enumerate(tqdm(testLoader)):
for i, (data, btag) in enumerate(tqdm(testLoader)):
    
    data = data.to(device)
    label = data.y.float().to(device=device)
    scale = data.ss.float().to(device)
    weight = data.ww.float().to(device)
    scaledweight = weight*scale
#     scaledweight = torch.abs(scaledweight)
    
    
    pred = model(data)
  
#     print(label)
#     print(data.pos)
#     stop
    btags.extend([x.item() for x in btag.view(-1)])
    labels.extend([x.item() for x in label])
    weights.extend([x.item() for x in weight])
    preds.extend([x.item() for x in pred.view(-1)])
    scaledWeights.extend([x.item() for x in (scaledweight).view(-1)])
    
    poss.extend([x.item() for x in data.pos.view(-1)])
    features.extend([x.item() for x in data.x.view(-1)])
    batch_size.append(data.x.shape[0])
    
    
#     procIdxs.extend([x.item() for x in procIdx])
#     fileIdxs.extend([x.item() for x in fileIdx])
#     idxs.extend([x.item() for x in idx])

# df = pd.DataFrame({'label':labels, 'prediction':preds, 'weight':weights, 'procIdx':procIdxs, 'fileIdx':fileIdxs, 'idx':idxs})

# fPred = 'result/' + args.output + '/' + args.output + '.csv'
# df.to_csv(fPred, index=False)

df = pd.DataFrame({'label':labels, 'prediction':preds,
                 'weight':weights, 'scaledWeight':scaledWeights})
fPred = 'result/' + args.output + '/' + args.output + '.csv'
df.to_csv(fPred, index=False)

df2 = pd.DataFrame({'pos':poss})
fPred2 = 'result/' + args.output + '/' + args.output + '_pos.csv'
df2.to_csv(fPred2, index=False)

df3 = pd.DataFrame({'feature':features})
fPred3 = 'result/' + args.output + '/' + args.output + '_feature.csv'
df3.to_csv(fPred3, index=False)

df4 = pd.DataFrame({'batch':batch_size})
fPred4 = 'result/' + args.output + '/' + args.output + '_batch.csv'
df4.to_csv(fPred4, index=False)

df5 = pd.DataFrame({'btag':btags})
fPred5 = 'result/' + args.output + '/' + args.output + '_btag.csv'
df5.to_csv(fPred5, index=False)
# from sklearn.metrics import roc_curve, roc_auc_score
# df = pd.read_csv(predFile)
# df2 = pd.read_csv(predonlyFile)







# ##### Draw ROC curve #####
# from sklearn.metrics import roc_curve, roc_auc_score
# df = pd.read_csv(fPred)
# tpr, fpr, thr = roc_curve(df['label'], df['prediction'], sample_weight=df['weight'], pos_label=0)
# auc = roc_auc_score(df['label'], df['prediction'], sample_weight=df['weight'])


# df_bkg = df[df.label==0]
# df_sig = df[df.label==1]

# hbkg1 = df_bkg['prediction'].plot(kind='hist', histtype='step', weights=df_bkg['weight'], bins=np.linspace(0, 1, 50), alpha=0.7, color='red', label='Fast Neutron')
# hsig1 = df_sig['prediction'].plot(kind='hist', histtype='step', weights=df_sig['weight'], bins=np.linspace(0, 1, 50), alpha=0.7, color='blue', label='Michel electrons')
# #plt.yscale('log')
# plt.ylabel('Events')
# plt.legend(loc = 'upper center')
# plt.savefig('result/' + args.output + '/' + args.output + '_Events.png', dpi=300)
# #plt.show()
# plt.clf()

# plt.plot(fpr, tpr, '.-', label='%s %.3f' % (args.output, auc))
# plt.xlabel('FN efficiency')
# plt.ylabel('ME efficiency')
# #plt.xlim(0, 0.001)
# plt.xlim(0, 1.000)
# plt.ylim(0, 1.000)
# plt.legend(loc = 'lower center')
# plt.savefig('result/' +args.output + '/' + args.output + '_efficiency.png', dpi=300)
# plt.grid()
# #plt.show()
# plt.clf()

