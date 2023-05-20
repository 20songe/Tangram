import h5py
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import csc_matrix
import scipy.io
import anndata
import tangram as tg
import seaborn as sns
from datetime import datetime
import pytz

def read_data():
    """
    Reads in the spatial and scRNAseq data and saves them as AnnData objects

    Returns: ad_sc, the scRNAseq AnnData object
             ad_sp, the spatial AnnData object
    """
    ad_sc = sc.read_10x_mtx('../cytospace/data')
    ad_sc.var_names_make_unique(join="-")
    ad_sc.obs_names_make_unique(join="-")
    meta = pd.read_csv('../cytospace/data/Meta_GBM.txt', header=0)
    meta_df = pd.DataFrame(meta)
    meta_df = meta_df[1:]
    ad_sc.obs = meta_df

    ad_sp = sc.read_visium('/users/esong18/data/Visium_2021/30-480463055/00_fastq/hires_results/DF-L1_S1/outs',
                             count_file='/users/esong18/data/Visium_2021/30-480463055/00_fastq/hires_results/DF-L1_S1/outs/filtered_feature_bc_matrix.h5',
                             source_image_path = '/users/esong18/data/Visium_2021/30-480463055/00_fastq/hires_results/DF-L1_S1/outs/spatial/tissue_hires_image.png')
    
    return ad_sc, ad_sp

def train(ad_sc, ad_sp):
    """
    Trains the Tangram model
    """
    markers_df = pd.DataFrame(ad_sc.var.index).iloc[0:500, :]
    markers = list(np.unique(markers_df.melt().value.values))

    tg.pp_adatas(ad_sc, ad_sp, genes=markers)
    ad_map = tg.map_cells_to_space(ad_sc, ad_sp, mode="cells", device = "cuda:0")

    return ad_map

def convert_to_h5ad(ad_map, name):
    ad_map.X = csc_matrix(ad_map.X)
    ad_map.write_h5ad(name)

def main():
    ad_sc, ad_sp = read_data()
    print("scRNAseq data: ", ad_sc)
    print("obs: ", ad_sc.obs)
    print("var: ", ad_sc.var)
    print("spatial data: ", ad_sp)
    ad_map = train(ad_sc, ad_sp)
    # convert_to_h5ad(ad_map, "saved_admap")
    print("ad_map is made: ", ad_map)
    ad_ge = tg.project_genes(ad_map, ad_sc)
    print("ad_ge: ", ad_ge)


    tg.plot_training_scores(ad_map, bins=20, alpha=.5)

    print("ad map uns: ", ad_map.uns['train_genes_df'])
    
    df_all_genes = tg.compare_spatial_geneexp(ad_ge, ad_sp, ad_sc)
    print(df_all_genes.head())

    tg.plot_auc(df_all_genes)
    print("plotted auc")

    # plot cell annotation
    tg.project_cell_annotations(ad_map, ad_sp, annotation="SubCluster")
    print("projected cell annotations: admap-", ad_map)
    print("ad_sp: ", ad_sp)
    annotation_list = list(pd.unique(ad_sc.obs["SubCluster"]))
    tg.plot_cell_annotation_sc(ad_sp, annotation_list,x='x', y='y',perc=0.001)
    
    tz_NY = pytz.timezone('America/New_York')
    now = datetime.now(tz_NY)
    print("process completed at: ", now.strftime("%H:%M:%S"))

if __name__ == "__main__":
    main()
