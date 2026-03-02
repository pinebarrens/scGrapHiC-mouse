import enum
import os


import numpy as np
import pandas as pd



# from src.globals import *
from src.utils import create_directory
from src.preprocess_datasets import read_cell_by_gene_matrix, read_pairix_file

def pseudobulk_rnaseq(tissue, cell_names, cell_type, stage=None):
    umi_file = os.path.join(
        "/users/mliu237/scratch/LiMCA/", 
        '/users/mliu237/scratch/LiMCA/GSE239969_LiMCA_olfactory_rna_gene_counts.tsv'
    )
    cell_names_with_gene = ['gene'] + cell_names
    umi_data = read_cell_by_gene_matrix(umi_file)
    umi_data = umi_data.filter(cell_names_with_gene)
    umi_data[cell_type] = umi_data[cell_names].sum(axis=1)
    umi_data = umi_data.drop(cell_names, axis=1)
    umi_data[cell_type] = umi_data[cell_type]/len(cell_names)
    
    if stage:
        output_path = os.path.join(
            "/users/mliu237/scratch/LiMCA/raw/pseudobulk/RNAseq",
            '{}_{}_{}_n{}_umi.csv.gz'.format(stage, tissue, cell_type, len(cell_names))
        )
        
    else:
        output_path = os.path.join(
            "/users/mliu237/scratch/LiMCA/raw/pseudobulk/RNAseq",
            '{}_{}_n{}_umi.csv.gz'.format(tissue, cell_type, len(cell_names))
        )
    
    print('Saving pseudobulk UMI file at {}'.format(output_path))
    umi_data.to_csv(output_path, index=False)
    
    

def pseudobulk_schic(tissue, cell_names, cell_type, stage=None):

        # Path to the single .pairs.gz file containing all contacts
        schic_file = "/users/mliu237/scratch/LiMCA/GSE239969_merged_pairs/allValidPairs_concat.txt"
        # TODO: Update the above path if the filename is different

        # Read the full file into a DataFrame
        schic_df = read_pairix_file(schic_file)
        # Filter for only the cell barcodes of interest
        schic_df = schic_df[schic_df['readID'].isin(cell_names)]

        # Optionally, you can print the number of contacts for debug
        print(f"Total contacts for {cell_type}: {len(schic_df)}")

        if stage:
            output_path = os.path.join(
                "/users/mliu237/scratch/LiMCA/raw/pseudobulk/scHiC",
                '{}_{}_{}_n{}_schic.pairs'.format(stage, tissue, cell_type, len(cell_names))
            )
        else:
            output_path = os.path.join(
                "/users/mliu237/scratch/LiMCA/raw/pseudobulk/scHiC",
                '{}_{}_n{}_schic.pairs'.format(tissue, cell_type, len(cell_names))
            )

        print('Saving pseudobulk scHi-C file at {}'.format(output_path))
        schic_df.to_csv(output_path, index=False, header=False, sep='\t')



def parse_metadata(metadata):

    try:
        cell_types = metadata['celltype'].unique().tolist()
    except KeyError:
        cell_types = metadata['celltype'].unique().tolist()
    cell_names = []
    
    for cell_type in cell_types:
        try:
            cell_name = metadata[(metadata['celltype'] == cell_type)]['DNAbarcode'].tolist()
        except KeyError:
            cell_name = metadata[(metadata['celltype'] == cell_type)]['DNAbarcode'].tolist()

        cell_names.append(cell_name)
    return cell_types, cell_names


def create_pseudobulk_files(path):
    metadata = pd.read_csv(path, sep='\t')
    tissue = "olfactory"

    if tissue == 'olfactory':
        # we repeat the process for all stages
        stages = metadata['celltype'].unique().tolist()
        for stage in stages: 
            cell_types, cell_names = parse_metadata(metadata[(metadata['celltype'] == stage)])
            for i, cell_type in enumerate(cell_types):
                pseudobulk_rnaseq(tissue, cell_names[i], cell_type, stage)
                pseudobulk_schic(tissue, cell_names[i], cell_type, stage)
    
    """
    elif tissue == 'brain':
        # there is only one type of cells
        cell_types, cell_names = parse_metadata(metadata)
        for i, cell_type in enumerate(cell_types):
            pseudobulk_rnaseq(tissue, cell_names[i], cell_type)
            pseudobulk_schic(tissue, cell_names[i], cell_type)
            

    else:
        print('Invalid metadata file path, exiting program...')
        exit(1)   
    """
create_pseudobulk_files("/users/mliu237/scratch/LiMCA/GSE239969_LiMCA_metadata.txt")
    
















