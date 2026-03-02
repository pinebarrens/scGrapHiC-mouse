import os
import time

import cooler
import hicstraw

import scanpy as sc
import pandas as pd
import lightning.pytorch as pl

from src.utils import *
from src.globals import *
from anndata import AnnData
from multiprocessing import Process
from scipy.sparse import csr_matrix
from src.visualizations import visualize_hic_contact_matrix, visualize_scnrna_seq_tracks

def read_pairix_file(path):
    '''
        This function reads a pairix file format (.pairs) and returns 
        a dictionary of numpy arrays
        @params: <string> - path, path to the file 
        @returns: <pd.DataFrame> - pandas dataframe 
    '''
    if os.path.exists(path):
        data = pd.read_csv(
            path, header = None,
            comment ='#', delim_whitespace=True, 
            names=[
                'readID', 
                'chr1', 'pos1',
                'chr2', 'pos2',
                'strand1', 'strand2',
                'phase0', 'phase1'
            ]
            # dtype={
            #     "readID": str, 
            #     "chr1": str, "pos1": int,
            #     "chr2": str, "pos2": int,
            #     "strand1": str, "strand2": str,
            #     'phase0': str , 'phase1': str - modified to match Droplet data
            # }

        )
        # print(data['readID'])
        # print(data['chr1'])
        # print(data['pos1'])
        # print(data['chr2'])
        # print(data['pos2'])
        # print(data['strand1'])
        # print(data['strand2'])
        data['pos1'] = pd.to_numeric(data["pos1"])
        data['pos2'] = pd.to_numeric(data["pos2"])

        return data    


def convert_pairs_to_pixels(dataframe, PARAMETERS):
    dataframe.loc[:, 'pos1'] = dataframe['pos1'].copy().floordiv(PARAMETERS['resolution'])
    dataframe.loc[:, 'pos2'] = dataframe['pos2'].copy().floordiv(PARAMETERS['resolution'])
    pixels = dataframe.groupby(['pos1', 'pos2']).size().reset_index(name='counts')
    pixels = pixels.rename(columns={'pos1': 'bin1_id', 'pos2': 'bin2_id', 'counts': 'count'}) 
    return pixels

  

            
def parse_hires_schic_datasets(input_path, PARAMETERS, output_path):
    chrom_sizes = read_chromsizes_file(os.path.join('/users/mliu237/scratch/LiMCA/raw/', 'chrom.sizes'))
    schic_files = list(map(lambda x: os.path.join(input_path, x), os.listdir(input_path)))            
    
    
    for schic_file in schic_files:
        if '.pairs' in schic_file and os.path.exists(schic_file):
            # Step 0: create the output directory
            cell_name = schic_file.split('/')[-1].split('.')[0]
            output_directory = os.path.join(output_path, cell_name)
            create_directory(output_directory)
            
            # Step 1: parse the .pairs.gz file 
            pairs_data = read_pairix_file(schic_file)
            pairs_data = pairs_data.drop(['readID', 'strand1', 'strand2'], axis=1) # , phase0, phase1 removed - modified to match Droplet data
            
            
            for chromosome, size in chrom_sizes.items():
                
                output_cooler_file = os.path.join(output_directory, '{}_{}.cool'.format(chromosome, PARAMETERS['resolution']))
                output_numpy_file = os.path.join(output_directory, '{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
                
                chrom_data = pairs_data.loc[(pairs_data['chr1'] == chromosome) & (pairs_data['chr2'] == chromosome)]
                chrom_pixels = convert_pairs_to_pixels(chrom_data, PARAMETERS)
                bins = chrom_bins(chromosome, size, PARAMETERS['resolution'])

                cooler.create_cooler(output_cooler_file, bins, chrom_pixels,
                                dtypes={"count":"int"}, 
                                assembly="mm10")
                
                # Read and normalize and save as a npy file
                clr = cooler.Cooler(output_cooler_file)
                matrix = clr.matrix(balance=False).fetch(chromosome)

                #  Saving the entire matrix 
                print('Saving: {}'.format(output_numpy_file))
                np.save(output_numpy_file, matrix)
                
PARAMETERS = initialize_parameters_from_args()
pl.seed_everything(PARAMETERS['seed'])
print(PARAMETERS)

parse_hires_schic_datasets("/users/mliu237/scratch/LiMCA/raw/pseudobulk/scHiC", PARAMETERS, "/users/mliu237/scratch/LiMCA/preprocessed/pseudobulk/scHiC")