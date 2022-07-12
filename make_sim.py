#!/usr/bin/env python3

import os
import os.path as osp
import argparse as arp
from typing import Dict,Callable,List

import pandas as pd
import numpy as np
import torch as t
import torch.distributions as dists
import scanpy as sc
from progress.bar import Bar
import time

import warnings
warnings.filterwarnings("ignore")


def _assemble_spot(cnt,
                  labels,
                  scale,
                  bounds : List[int] = [10,30],
                  )->Dict[str,t.Tensor]:

    """Assemble single spot

    generates one synthetic ST-spot
    from provided single cell data

    Parameter:
    ---------
    cnt : np.ndarray
        single cell count data [n_cells x n_genes]
    labels : np.ndarray
        single cell annotations [n_cells]
    alpha : float
        dirichlet distribution
        concentration value
    fraction : float
        fraction of transcripts from each cell
        being observed in ST-spot

    Returns:
    -------
    Dictionary with expression data,
    proportion values and number of
    cells from each type at every
    spot

    """

    # sample cells to be present
    # at spot
    n_cells = dists.uniform.Uniform(low = bounds[0],
                                    high = bounds[1]).sample().round().type(t.int)
    # get unique labels found in single cell data
    uni_labs, uni_counts = np.unique(labels.values,
                                     return_counts = True)

    # make sure sufficient number
    # of cells are present within
    # all cell types
    assert np.all(uni_counts >= 5), \
            "Insufficient number of cells"

    # get number of different
    # cell types present
    n_labels = uni_labs.shape[0]

    # sample number of types to
    # be present at current spot
    high = t.tensor(min(n_cells, n_labels), dtype=t.float)
    n_types = dists.uniform.Uniform(low = 1,
                                    high = high+1).sample()
    #
    n_types = n_types.round().type(t.int)

    # select which types to include
    pick_types = t.randperm(n_labels)[0:n_types]
    # pick at least one cell for spot
    members = t.zeros(n_labels).type(t.float)
    while members.sum() < 1:
        # draw proportion values from probability simplex
        member_props = dists.Dirichlet(concentration = 1.0 * t.ones(n_types)).sample()
        # get integer number of cells based on proportions
        members[pick_types] = (n_cells * member_props).round()
    # get proportion of each type
    props = members / members.sum()
    # convert to ints
    members = members.type(t.int)
    # get number of cells from each cell type

    # generate spot expression data
    spot_expr = t.zeros(cnt.n_vars).type(t.float32)

    for z in range(n_types):
        # get indices of selected type
        idx = np.where(labels == uni_labs[pick_types[z]])[0]
        # pick random cells from type
        np.random.shuffle(idx)
        idx = idx[0:members[pick_types[z]]]
        # add fraction of transcripts to spot expression
        spot_expr +=  t.tensor((cnt.X[idx].todense()).sum(axis = 0).astype(np.float32)).squeeze(dim=0)
    spot_expr = np.round(spot_expr)
    # spot_expr/=members.sum()
    # add noise
    noise_vec = np.random.normal(loc=0, scale=scale, size=spot_expr.shape)
    spot_expr = t.tensor(spot_expr + noise_vec, dtype=t.float)
    spot_expr = t.nn.functional.relu(spot_expr)

    return {'expr':spot_expr,
            'proportions':props,
            'members': members,
           }


def assemble_data_set(cnt,
                      n_spots : int,
                      n_cell_range : List[int],
                      scale : float,
                      assemble_fun : Callable = _assemble_spot,
                     )-> Dict[str,pd.DataFrame]:

    """Assemble Synthetic ST Data Set

    Assemble synthetic ST Data Set from
    a provided single cell data set

    Parameters:
    ----------
    cnt : pd.DataFrame
        single cell count data
    labels : pd.DataFrame
        single cell annotations
    n_spots : int
        number of spots to generate
    n_genes : int
        number of gens to include
    assemble_fun : Callable
        function to assemble single spot

    """

    # get labels
    labels = cnt.obs['label']

    # get unique labels
    uni_labels = np.unique(labels.values)
    n_labels = uni_labels.shape[0]

    # prepare matrices
    st_cnt = np.zeros((n_spots,cnt.n_vars))
    st_prop = np.zeros((n_spots,n_labels))
    st_memb = np.zeros((n_spots,n_labels))



    # generate one spot at a time
    bar = Bar('Generate pseudo spots from scRNA-seq: ', max=n_spots)
    bar.check_tty = False
    for spot in range(n_spots):
        spot_data = assemble_fun(cnt,
                                 labels,
                                 bounds = n_cell_range,
                                 scale = scale
                                 )

        st_cnt[spot,:] = spot_data['expr']
        st_prop[spot,:] = spot_data['proportions']
        st_memb[spot,:] =  spot_data['members']
        # st_regi[spot, :] = spot_data['region']

        index = pd.Index(['Spotx' + str(x + 1) for \
                          x in range(n_spots) ])
        bar_str = '{} / {}'
        bar.suffix = bar_str.format(spot + 1, n_spots)
        bar.next()
    bar.finish()

    # columns = [s.decode('utf-8') for s in cnt.var_names.values.tolist()]

    # convert to scanpy Anndata

    st_cnt = sc.AnnData(st_cnt)
    st_cnt.obs_names = index.values
    st_cnt.var_names = cnt.var_names

    st_cnt.uns['proportions'] = pd.DataFrame(st_prop,
                           index = index,
                           columns = uni_labels,
                          )
    st_cnt.uns['member'] = pd.DataFrame(st_memb,
                           index = index,
                           columns = uni_labels,
                           )

    return st_cnt


def main():

    prs = arp.ArgumentParser()

    prs.add_argument('-c','--sc_counts',
                     type = str,
                     default='data/DLPFC/sc_DLPFC.h5ad',
                     help = 'path to single cell annotation data.'
                            'Cell type name in .obs (default: label)'
                     )

    prs.add_argument('-ns','--n_spots',
                     type = int,
                     default = 10000,
                     help = 'number of spots',
                    )

    prs.add_argument('-nosie','--std',
                     type = int,
                     default = 0,
                     help = 'std of noise added',
                    )

    prs.add_argument('-o','--out_dir',
                     default = 'data/DLPFC',
                     help = 'output directory',
                    )

    prs.add_argument('-ncr','--n_cell_range',
                     nargs = 2,
                     default = [1,10],
                     type = int,
                     help = 'lower bound (first argument)'\
                     " and upper bound (second argument)"\
                     " for the number of cells at each spot",
                    )


    prs.add_argument('-t','--tag',
                     default = '10000_DLPFC_1_10',
                     help = 'tag to mark data se with',
                    )

    args = prs.parse_args()

    if args.out_dir is None:
        out_dir = osp.dirname(args.sc_counts)
    else:
        out_dir = args.out_dir

    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    sc_cnt_pth =  args.sc_counts

    n_spots = args.n_spots

    sc_cnt = sc.read_h5ad(sc_cnt_pth)
    sc_cnt.var_names_make_unique()

    sc.pp.filter_cells(sc_cnt, min_genes=200)
    sc.pp.filter_genes(sc_cnt, min_cells=3)
    id_tmp1 = np.asarray([not str(name).startswith('ERCC') for name in sc_cnt.var_names], dtype=bool)
    id_tmp2 = np.asarray([not str(name).startswith('MT-') for name in sc_cnt.var_names], dtype=bool)
    id_tmp = np.logical_and(id_tmp1, id_tmp2)
    sc_cnt._inplace_subset_var(id_tmp)
    sc.pp.normalize_total(sc_cnt, target_sum=10000)


    sim = assemble_data_set(sc_cnt,
                                      n_spots = n_spots,
                                      n_cell_range = args.n_cell_range,
                                      assemble_fun = _assemble_spot,
                                      scale = args.std
                                      )
    sim.write_h5ad(osp.join(out_dir, '.'.join(['pseudo',args.tag,'h5ad'])))


if __name__ == '__main__':
    main()
