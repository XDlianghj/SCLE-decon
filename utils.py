import numpy as np
import pandas as pd
from sklearn import neighbors
import logging
import scanpy as sc
import os
from scipy.spatial.distance import pdist,squareform, cdist
from scipy.spatial import KDTree
import scipy.sparse as sp
import random


def adata_preprocess(i_adata, a_adata, markers, min_cells=3):
    '''Select genes and normalized'''
    print('===== Preprocessing Data ')
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    sc.pp.normalize_total(a_adata, target_sum=1)
    sc.pp.normalize_total(i_adata, target_sum=1)

    keep = [val for val in markers if val in i_adata.var.index]

    i_adata = i_adata[:, keep]
    a_adata = a_adata[:, keep]

    sc.pp.scale(a_adata)
    sc.pp.scale(i_adata)
    return i_adata, a_adata


def trans(index_names, columns_names, path, threshold=0.02):
    res = pd.read_csv(path, header=None)
    df = pd.DataFrame(res.values, columns=columns_names, index=index_names)
    df[df < threshold] = 0
    df = df.div(df.sum(axis=1), axis=0)
    df.to_csv(path)


def top_n_idx_sparse(matrix, n):
    '''Return index of top n values in each row of a sparse matrix'''
    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        top_n_idx.append(matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]])
    return top_n_idx


def draw_batch(batch1, batch2, label1, label2):
    '''draw diffidence of two batch and calculate KL divergence'''
    ann_real = sc.AnnData(batch1)
    ann_fake = sc.AnnData(batch2)
    ann_fake.obs_names = label1.index.values
    ann_real.obs_names = label2.index.values
    ann_fake.obs[label1.columns.values] = label1
    ann_real.obs[label2.columns.values] = label2
    # print(label1)
    ann_fake.obs['label'] = pd.Categorical(np.argmax(label1.values,axis=1))
    ann_real.obs['label'] = pd.Categorical(np.argmax(label2.values,axis=1))
    new = ann_fake.concatenate(ann_real, join='inner')
    # new.obs['label'] = pd.Categorical(new.obs['label'])
    sc.tl.pca(new, svd_solver='arpack')
    # sc.pl.pca(new, color='batch')

    sc.pp.neighbors(new,n_neighbors=25)
    ng = new.obsp['connectivities']
    neighbors = top_n_idx_sparse(ng, 20)
    qb0 = new.obs['batch'].value_counts()[0] / new.n_obs
    qb1 = new.obs['batch'].value_counts()[1] / new.n_obs
    kl_res = []
    for _ in range(20):
        slice = random.sample(range(10000, 13000), 100)
        ans = []
        n0=0
        n1=0
        for i in slice:
            neigh = new[neighbors[i], :]
            new1 = neigh[neigh.obs['label']==new[i].obs['label'].values[0]]
            if new1.n_obs==0:
                continue
            pb0 = 1 - sum(list(map(int, new1.obs['batch'].tolist()))) / new1.n_obs
            pb1 = 1-pb0
            if pb0==0:
                n0+=1
                kl = pb1 * np.log(pb1 / qb1)
            elif pb1==0:
                n1+=1
                kl = pb0 * np.log(pb0 / qb0)
            else:
                kl = pb0 * np.log(pb0 / qb0) + pb1 * np.log(pb1 / qb1)
            ans.append(kl)
        kl_res.append(np.mean(ans))

    sc.tl.tsne(new)
    sc.pl.tsne(new, color=['batch'])
    sc.pl.tsne(new, color=['label'])
    sc.pl.tsne(new,
               color=['Astrocytes', 'Bergmann', 'Choroid', 'Endothelial', 'Granule', 'Microglia', 'Oligodendrocytes',
                      'Purkinje'])

    return kl_res


def FindMarkers(Anndata, num_markers):
    '''Find markers genes of each cluster from scRNA-seq'''
    print('===== Finding markers genes ')
    Anndata.obs['label'] = Anndata.obs['label'].astype('category', copy=False)
    sc.tl.rank_genes_groups(Anndata, 'label', method='wilcoxon')
    genelists=Anndata.uns['rank_genes_groups']['names']
    df_genelists = pd.DataFrame.from_records(genelists)

    # Combining top marker genes representing each cell type
    res_genes = []
    for column in df_genelists.head(num_markers):
        res_genes.extend(df_genelists.head(num_markers)[column].tolist())
    res_genes_ = list(set(res_genes))
    res_genes_.sort(key=res_genes.index)
    return res_genes_


def FindNeighbours(st_loc_df,  k_cutoff=None, max_neigh=50, label=None):
    '''Calculating neighbor graph'''
    print('===== Calculating neighbor graph ')
    nbrs = neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(st_loc_df)
    distances, indices = nbrs.kneighbors(st_loc_df)
    indices = indices[:, 0:k_cutoff + 1]
    distances = distances[:, 0:k_cutoff + 1]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    if label is not None:
        pro_labels_dict = dict(zip(range(label.shape[0]), label))
        KNN_df['Cell1_label'] = KNN_df['Cell1'].map(pro_labels_dict)
        KNN_df['Cell2_label'] = KNN_df['Cell2'].map(pro_labels_dict)
        KNN_df = KNN_df.loc[KNN_df['Cell1_label'] == KNN_df['Cell2_label'],]
    G = sp.coo_matrix((np.ones(KNN_df.shape[0]), (KNN_df['Cell1'], KNN_df['Cell2'])), shape=(len(st_loc_df), len(st_loc_df)))
    return G
