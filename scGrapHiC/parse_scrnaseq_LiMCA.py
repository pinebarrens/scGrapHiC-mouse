'''
    This file contains the functions to preprocess the RNAseq dataset
'''
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


# Assume this is the same for the human gene annotation file
def get_gene_name(attributes):
    '''
        Extracts the gene name from the GTF3 file attributes dictionary
    '''
    attributes = attributes.split(';')
    attribute = list(filter(lambda x: 'gene_name' in x, attributes))[0]
    attribute = attribute.split('=')[-1]
    return attribute



def process_gtf3_file(gtf3_filepath, output_path):
    '''
        Process the GTF3 file to get the geneome coordinate maping from the gene names, 
        its a supporting file required for processing RNA-seq UMI cell-by-gene matrices
    '''

    data = pd.read_csv(
        gtf3_filepath, header = None,
        comment ='#', sep ='\t', 
        names=[
            'seqid', 
            'source', 'type',
            'start', 'end', 
            'score', 'strand', 'phase', 
            'attributes'
        ]
    )
    # We only need the genes annotations
    data = data.loc[data['type'] == 'gene']
    # We only need the gene name
    data['attributes'] = data['attributes'].map(get_gene_name)

    data = data.drop(columns=['source', 'type', 'score', 'phase'])
    data = data.rename(columns={'seqid': 'chr', 'attributes': 'gene_name'})
    
    print('Created the Gene Coordinate file, saving it at: {}'.format(output_path))
    data.to_csv(output_path, index=False)


def read_cell_by_gene_matrix(path):
    format = path.split('/')[-1].split('.')[-2]
    
    if format == 'csv':
        sep = ','
    elif format == 'tsv':
        sep = '\t'
    else:
        sep = None
    
    cell_by_gene_data = pd.read_csv(
        path, sep=sep,
        comment='#'
    )
    return cell_by_gene_data

def normalize_cell_by_gene_matrix(cell_by_gene_matrix):
    X = cell_by_gene_matrix.iloc[1:, 1:].to_numpy()
    genes = cell_by_gene_matrix.iloc[1:, 0:1].to_numpy().reshape(-1)
    cells = cell_by_gene_matrix.columns[1:].to_numpy()
    
    adata = AnnData(X.T)
    X_norm = sc.pp.normalize_total(adata, target_sum=1, inplace=False)['X']
    X_norm = X_norm.T
    cell_by_gene_matrix = pd.DataFrame(data=X_norm, index=genes, columns=cells)
    cell_by_gene_matrix = cell_by_gene_matrix.reset_index().rename(columns={'index':'gene'})	
    
    return cell_by_gene_matrix

def convert_cell_by_gene_to_coordinate_matrix(cell_by_gene_matrix, gene_coordinate_file):
    gene_coordinates = pd.read_csv(gene_coordinate_file)
    # print(cell_by_gene_matrix)
    # print(gene_coordinates)
    print("Columns:", cell_by_gene_matrix.columns.tolist())
    print(cell_by_gene_matrix.head())
    # Left join on both tables and we drop the gene_names and have NaNs at genes that were filtered, we replace them with 0s. 
    merged_tables = pd.merge(gene_coordinates, cell_by_gene_matrix, left_on='gene_name', right_on='gene', how='left').drop('gene_name', axis=1)
    # print(merged_tables)
    return merged_tables


def create_coordinate_matrix_file(scrna_seq_file, gene_cooridate_file_path, output_folder, PARAMETERS):
    scrna_seq_file_name = scrna_seq_file.split('/')[-1].split('.')[0]
    output_intermediate_file = os.path.join(output_folder, scrna_seq_file_name + '.csv')
    
    if os.path.exists(output_intermediate_file):
        print('Coordinate matrix file {} already exists'.format(output_intermediate_file))
        return output_intermediate_file
    
    # Step 1: read the file into a pandas dataframe
    cell_by_gene_data = read_cell_by_gene_matrix(scrna_seq_file)

    # Step 1.5: normalize the cell-by-gene matrix file 
    if PARAMETERS['normalize_umi']:
        cell_by_gene_data = normalize_cell_by_gene_matrix(cell_by_gene_data)
    
    # Step 2: convert it into a gene coordinate format
    coordinate_matrix = convert_cell_by_gene_to_coordinate_matrix(cell_by_gene_data, gene_cooridate_file_path)
    
    # Checkpoint here, and store the coordinate matrix file
    print('Saving coordinate matrix file {}'.format(output_intermediate_file))
    coordinate_matrix.to_csv(output_intermediate_file)
    
    
    return output_intermediate_file
    
    
    
    
def merge_chr_coordinates(coordinates):
    # Aggregate the UMI reads based on the starting and ending coordinate
    coordinates = coordinates.groupby(['start', 'end']).sum()
    # Drop some useless rows
    coordinates = coordinates.drop(['chr', 'strand', 'gene'], axis=1)
    coordinates = coordinates.reset_index()

    return coordinates


def normalize_genomic_track(reads, PARAMETERS):
    sum_reads = np.sum(reads)
    reads = np.divide(reads, sum_reads)
    reads = reads * PARAMETERS['library_size']
    reads = np.log1p(reads)
    reads = reads/np.max(reads)
    
    return reads
    

def create_genomic_track(starts, ends, reads, chrsize, PARAMETERS):
    size = chrsize // PARAMETERS['resolution'] + 1
    
    track = np.zeros(size)
    
    for i in range(starts.shape[0]):
        track[starts[i]:ends[i] + 1] += reads[i]/((ends[i] + 1) - starts[i])
    
    if PARAMETERS['normalize_track']:
        track = normalize_genomic_track(track, PARAMETERS)
    
    return track



def create_genomic_track_file(preprocessed_coordinate_matrix, PARAMETERS, output_path, pseudobulk=False):
    chrom_sizes = read_chromsizes_file(os.path.join("/users/mliu237/scratch/LiMCA/raw", 'chrom.sizes'))    
    coordinate_matrix = pd.read_csv(preprocessed_coordinate_matrix)
    coordinate_matrix = coordinate_matrix.loc[:, ~coordinate_matrix.columns.str.contains('^Unnamed')]
    
    # Step 3: proces this coordinate matrix with pandas operations
    # Step 3.1: Replace NaNs with zeros
    coordinate_matrix.dropna(subset=['gene'], inplace=True)
    coordinate_matrix = coordinate_matrix.fillna(0)
    
    # Step 3.2: Convert coordinate scale in accordance with the resolution
    coordinate_matrix['start'] = coordinate_matrix['start'].copy().floordiv(PARAMETERS['resolution'])
    coordinate_matrix['end'] = coordinate_matrix['end'].copy().floordiv(PARAMETERS['resolution'])
    
    # Step 3.3: divide the positive and negative strands into two dataframes
    positive_strand_coordinates = coordinate_matrix.loc[coordinate_matrix['strand'] == '+']
    negative_strand_coordinates = coordinate_matrix.loc[coordinate_matrix['strand'] == '-']
    
    # Step 3.4: All cell types
    cells = list(coordinate_matrix.columns[5:])
    
    # Step 4: for each chromosome extract the track for each cell
    for chromosome, size in chrom_sizes.items():
        # step 4.1: extract reads only specific to a chromosome
        chr_positive_strand_coordinates = positive_strand_coordinates.loc[positive_strand_coordinates['chr'] == chromosome]
        chr_negative_strand_coordinates = negative_strand_coordinates.loc[negative_strand_coordinates['chr'] == chromosome]
        
        # step 4.2: merge the coordinates that overlap
        chr_positive_strand_coordinates = merge_chr_coordinates(chr_positive_strand_coordinates)
        chr_negative_strand_coordinates = merge_chr_coordinates(chr_negative_strand_coordinates)
        
        #step 4.3: for each cell extract the tracks
        for cell in cells:
            if pseudobulk:
                # For pseudo-bulking we have to store the other paramters of pseudobulking
                stage, tissue, cell_type, num_cells = get_file_name_parameters(preprocessed_coordinate_matrix)
                folder = '_'.join([stage, tissue, cell_type, 'n{}'.format(num_cells), 'scrnaseq']) if stage else  '_'.join([tissue, cell_type, 'n{}'.format(num_cells, 'scrnaseq')])
                output_folder = os.path.join(output_path, folder)
            else:
                stage, tissue, cell_type, num_cells = get_file_name_parameters(preprocessed_coordinate_matrix)
                folder = '_'.join([stage, tissue, cell_type, 'n{}'.format(num_cells), 'scrnaseq']) if stage else  '_'.join([tissue, cell_type, 'n{}'.format(num_cells, 'scrnaseq')])
                output_folder = os.path.join(output_path, folder)
            
            create_directory(output_folder)
            
            output_file = os.path.join(output_folder, '{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
            
            #step 4.4: extract both postive and negative tracks
            positive_track = create_genomic_track(
                chr_positive_strand_coordinates['start'].to_numpy(), 
                chr_positive_strand_coordinates['end'].to_numpy(), 
                chr_positive_strand_coordinates[cell].to_numpy(), 
                size,
                PARAMETERS
            )
            
            negative_track = create_genomic_track(
                chr_negative_strand_coordinates['start'].to_numpy(), 
                chr_negative_strand_coordinates['end'].to_numpy(), 
                chr_negative_strand_coordinates[cell].to_numpy(), 
                size,
                PARAMETERS
            )
            
            combined_tracks = np.stack((positive_track, negative_track))
            
            # save the tracks
            print('Saving: {}'.format(output_file))
            
            np.save(output_file, combined_tracks)



def parse_hires_scrnaseq_datasets(input_path, PARAMETERS, output_path, pseudobulk=False):
    '''
        This function parses the scRNA-seq datasets from cell-by-gene to geneome coordinate tracks
    '''
    
    #Step 0: Setting up auxiliary files 
    #Since its a mouse dataset aligned on mm10 assembly, we first create the mm10 gene_coordinate track
    gene_cooridate_file_path = os.path.join("/users/mliu237/scratch/LiMCA/preprocessed", 'gene_coordinates.csv')
    if not os.path.exists(gene_cooridate_file_path):
        process_gtf3_file(
            "/users/mliu237/scratch/LiMCA/raw/gencode.vM23.annotation.gff3",
            gene_cooridate_file_path
        )

    
    
    
    scrna_seq_files = list(map(lambda x: os.path.join(input_path, x), os.listdir(input_path)))
    
    preprocessed_coordinate_matrices = []
    # For all the cell-by-gene UMI matrix files we do
    for scrna_seq_file in scrna_seq_files:
        output_intermediate_file = create_coordinate_matrix_file(
            scrna_seq_file,
            gene_cooridate_file_path,
            output_path,
            PARAMETERS
        )
        preprocessed_coordinate_matrices.append(output_intermediate_file)
    
    
    for preprocessed_coordinate_matrix in preprocessed_coordinate_matrices:
        create_genomic_track_file(
            preprocessed_coordinate_matrix, 
            PARAMETERS, 
            output_path,
            pseudobulk
        )
"""
def preprocess_hires_datasets(PARAMETERS):
    '''
        This function parses both scRNA-seq and scHi-C datasets
    '''
    #parse_hires_scrnaseq_datasets(MOUSE_RAW_DATA_SCRNASEQ, PARAMETERS, MOUSE_PREPROCESSED_DATA_SCRNASEQ)
    parse_hires_scrnaseq_datasets(MOUSE_RAW_DATA_PSEUDO_BULK_SCRNASEQ, PARAMETERS, MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCRNASEQ, True)
    
    # parse_hires_schic_datasets(MOUSE_RAW_DATA_SCHIC, PARAMETERS, MOUSE_PREPROCESSED_DATA_SCHIC)
    parse_hires_schic_datasets(MOUSE_RAW_DATA_PSEUDO_BULK_SCHIC, PARAMETERS, MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCHIC)
    
    parse_bulk_datasets(PARAMETERS)
    parse_motifs_datasets(PARAMETERS)
"""

PARAMETERS = initialize_parameters_from_args()
pl.seed_everything(PARAMETERS['seed'])
print(PARAMETERS)

parse_hires_scrnaseq_datasets("/users/mliu237/scratch/LiMCA/raw/pseudobulk/RNAseq", PARAMETERS, "/users/mliu237/scratch/LiMCA/preprocessed/pseudobulk/RNAseq", True)
