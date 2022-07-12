import torch
import argparse
import warnings
import numpy as np
import pandas as pd
import anndata
from SCLE_train import SCLE_Train
from sklearn import metrics
import matplotlib.pyplot as plt
import scanpy as sc
from utils import *
import random
import os
import time


warnings.filterwarnings('ignore')

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)
torch.cuda.manual_seed(2022)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.cudnn_enabled = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('===== Using device: ' + device)

# ################ Parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=6, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
parser.add_argument('--cl_w1', type=float, default=0.5, help='Weight of CL loss.')
parser.add_argument('--cl_w2', type=float, default=0.1, help='Weight of CL loss.')
parser.add_argument('--dec_kl_w', type=float, default=1, help='Weight of DEC loss.')
parser.add_argument('--dec_cluster_n', type=int, default=7, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=14, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.005, help='DEC tol.')
parser.add_argument('--layers', type=list, default=[1024, 256, 64], help='Dim of SCLE encoder layers.')
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
parser.add_argument('--dropout', type=float, default=0.9, help='Dropout rate of data enhancement.')
parser.add_argument('--noise', type=float, default=None, help='Std of noise added to data')
parser.add_argument('--temperature', type=float, default=0.07, help='Temperature of CL loss.')
parser.add_argument('--cl_lr', type=float, default=1, help='Initial CL learning rate.')
parser.add_argument('--mask_percentage', type=float, default=0.9, help='Eval graph kN tol.')
parser.add_argument('--apply_mask_prob', type=float, default=0.5, help='Eval graph kN tol.')
# ______________ Eval clustering Setting ______________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')
parser.add_argument('--deconv', type=bool, default=True, help='Eval graph kN tol.')
parser.add_argument('--cls_lr', type=float, default=0.001, help='Eval graph kN tol.')
parser.add_argument('--cls_epochs', type=int, default=2000, help='Eval graph kN tol.')
parser.add_argument('--hidden', type=int, default=32, help='Eval graph kN tol.')
parser.add_argument('--prune', type=bool, default=True, help='Eval graph kN tol.')
parser.add_argument('--pre_resolution', type=float, default=0.2, help='Eval graph kN tol.')

params = parser.parse_args()
params.device = device

# ################ Path setting

# all DLPFC folder list
# proj_list = ['151507', '151508', '151509', '151510',
#              '151669', '151670', '151671', '151672',
#              '151673', '151674', '151675', '151676']
# proj_list = ['A1', 'B1', 'C1', 'D1',
#              'E1', 'F1', 'G2', 'H1']
# proj_list = ['0', '0.1', '0.5', '1',
#              '1.5', '2', '2.5', '3',
#              '3.5', '4', '4.5', '5']
proj_list = [ '151673']

train_path = 'data/DLPFC/counts.10000_DLPFC_1_10.h5ad'
sc_path = 'data/DLPFC/sc_DLPFC.h5ad'

# read pesudo ST data
train = sc.read_h5ad(train_path)
train.var_names_make_unique()
train.obs_names = [s.decode('utf-8') for s in train.obs_names.values]
train.var_names = [s.decode('utf-8') for s in train.var_names.values]
# read proportion of pesudo ST data
label = train.uns['proportions']

# read scRNA-seq
sc_cnt = sc.read_h5ad(sc_path)
sc_cnt.obs_names = [s.decode('utf-8') for s in sc_cnt.obs_names.values]
sc_cnt.var_names = [s.decode('utf-8') for s in sc_cnt.var_names.values]
# preprocess scRNA-seq and find marker genes of each cell type
sc.pp.filter_cells(sc_cnt, min_genes=200)
sc.pp.filter_genes(sc_cnt, min_cells=3)
sc.pp.normalize_total(sc_cnt, target_sum=1e4)
marker = FindMarkers(sc_cnt, 80)

for name in proj_list:
    # read ST data
    adata_h5 = sc.read_visium('data/DLPFC/151673')
    adata_h5.var_names_make_unique()

    adata_X, adata_Sim = adata_preprocess(adata_h5, train, marker, min_cells=3)
    print(adata_X)

    # k_lres1 = draw_batch(adata_X.X, adata_Sim.X, label, test_label)
    # draw_batch(adata_X.X, adata_Sim.X)

    if params.prune == True:
        sc.tl.pca(adata_X, svd_solver='arpack')
        sc.pp.neighbors(adata_X)
        sc.tl.louvain(adata_X, resolution=params.pre_resolution, key_added='expression_louvain_label')
        A = FindNeighbours(adata_h5.obsm['spatial'], k_cutoff=params.k, label=adata_X.obs['expression_louvain_label'])
    else:
        A = FindNeighbours(adata_h5.obsm['spatial'], k_cutoff=params.k)

    params.cell_num = adata_h5.shape[0]

    # ################## Model training
    st = time.time()
    scle_net = SCLE_Train(adata_X, params, A, X_sim=adata_Sim, X_label=label.values.astype(float))
    predict = scle_net.deconvolution_fit()
    # print(time.time()-st)
    z_X = scle_net.process(adata_X.X)
    z_Sim = scle_net.process(adata_Sim.X)
    # np.save('D:/lhj/study/Code/SCLE-master/data/DLPFC/sim.npy', adata_Sim.X)
    # np.save('D:/lhj/study/Code/SCLE-master/data/DLPFC/sp.npy', adata_X.X)
    # np.save('D:/lhj/study/Code/SCLE-master/data/DLPFC/z_sim.npy', z_Sim)
    # np.save('D:/lhj/study/Code/SCLE-master/data/DLPFC/z_sp.npy', z_X)
    # k_lres2 = draw_batch(z_X, z_Sim, label, test_label)

    # draw_batch(z_X, z_Sim)
    # predict = scle_net.deconvolution_fit()

    np.savetxt('data/DLPFC/151673/SCLE_predict_A.csv', predict, delimiter=',')
    trans(adata_X.obs_names.values, label.columns.values,
          'data/DLPFC/151673/SCLE_predict_A.csv')
