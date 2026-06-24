import os
import json
import multiprocessing

import numpy as np

from src.globals import *
from itertools import repeat
from multiprocessing import Pool
from src.graph_pe import graph_pe
from sklearn.model_selection import train_test_split
from src.normalizations import normalizations, smooth_adjacency_matrix
from src.normalizations import smooth_adjacency_matrix, generate_expected_contact_matrix
from src.utils import get_file_name_parameters, divide_signal, divide_matrix, add_dataset, compactM, create_directory




MULTIPROCESSING = True


def create_schires_dataset(PARAMETERS):
    scrnaseq_files = os.listdir(MOUSE_PREPROCESSED_DATA_SCRNASEQ)
    schic_files = os.listdir(MOUSE_PREPROCESSED_DATA_SCHIC)
    
    
    # find the union of these files and only work with them 
    cells = list(set(scrnaseq_files) & set(schic_files))
    chromosme = 'chr1'
    # For each cell create a graphish and then store it in a list that will eventually form our dataloader
    bulk_hic_file = os.path.join(MOUSE_PREPROCESSED_DATA_BULK, PARAMETERS['bulk_hic'], '{}_{}.npz'.format(chromosme, PARAMETERS['resolution']))
    bulk_hic_object = np.load(bulk_hic_file, allow_pickle=True)
    bulk_hic_data = bulk_hic_object['hic']
        
    node_features = []
    bulks = []
    targets = []
    ab_datas = []
    tad_datas = []
    
    for cell in cells:
        try:
            scrnaseq_file = os.path.join(MOUSE_PREPROCESSED_DATA_SCRNASEQ, cell, '{}_{}.npy'.format(chromosme, PARAMETERS['resolution']))
            scrnaseq_data = np.load(scrnaseq_file).transpose(1, 0)
            
            schic_file = os.path.join(MOUSE_PREPROCESSED_DATA_SCHIC, cell, '{}_{}.npy'.format(chromosme, PARAMETERS['resolution']))
            schic_data = np.load(schic_file)            
            # schic_data = matImpute(schic_data, 1, -1)
            
            ab_file = os.path.join(MOUSE_PREPROCESSED_DATA_SCHIC, cell, 'compartments_{}_{}.npy'.format(chromosme, PARAMETERS['resolution']))
            ab_data = np.load(ab_file)
            
            tad_file = os.path.join(MOUSE_PREPROCESSED_DATA_SCHIC, cell, 'TADs_{}_{}.npy'.format(chromosme, PARAMETERS['resolution']))
            tad_dat = np.load(tad_file)
            
        except:
            print('Cell is missing for one of the two modalities.')
            continue
            
        
        pe = graph_pe(bulk_hic_data, encoding_dim=PARAMETERS['pos_encodings_dim'])
        
        if PARAMETERS['pos_encodings_dim'] != 0:
            node_feature = np.concatenate((scrnaseq_data.transpose(1, 0), pe.transpose(1, 0)))
        else:
            node_feature = scrnaseq_data.transpose(1, 0)
            
        
        
        
        
        node_features.append(node_feature.reshape(1, node_feature.shape[0], node_feature.shape[1]))
        # bulks.append(bulk_hic_data.reshape(1, bulk_hic_data.shape[0], bulk_hic_data.shape[1]))
        targets.append(schic_data.reshape(1, schic_data.shape[0], schic_data.shape[1]))
        ab_datas.append(ab_data.reshape(1, ab_data.shape[0], ab_data.shape[1]))
        tad_datas.append(tad_dat.reshape(1, tad_dat.shape[0], tad_dat.shape[1]))
        
    node_features = np.concatenate(node_features)
    # bulks = np.concatenate(bulks)
    targets = np.concatenate(targets)
    ab_datas = np.concatenate(ab_datas)
    tad_datas = np.concatenate(tad_datas)
    
    # node_features_train, node_features_test, bulks_train, bulks_test, targets_train, targets_test = train_test_split(node_features, bulks, targets, test_size=0.15)
    node_features_train, node_features_test, ab_train, ab_test, tad_train, tad_test, targets_train, targets_test = train_test_split(node_features, ab_datas, tad_datas, targets, test_size=0.15)
    
    train_output_file = os.path.join(MOUSE_PROCESSED_DATA_HIRES, 'train.npz')
    
    np.savez_compressed(train_output_file, 
        node_features=node_features_train, 
        hic_targets=targets_train,
        ab_targets=ab_train,
        tad_targets=tad_train
    )
    
    test_output_file = os.path.join(MOUSE_PROCESSED_DATA_HIRES, 'test.npz')
    
    np.savez_compressed(test_output_file, 
        node_features=node_features_test, 
        hic_targets=targets_test,
        ab_targets=ab_test,
        tad_targets=tad_test
    )
    
    

dataset_partitions = {
    'train': [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19],
    'valid': [19],
    'test':  [7, 11],
    'ood':   [7, 11],
    'debug': [1]
}



def create_chromosome_dataset(rna_seq_dataset_path, schic_dataset_path, chromosome, PARAMETERS, bulk_hic_dir=None, motifs_dir=None):
    if not bulk_hic_dir:
        raise ValueError("bulk_hic_dir is required (pass --bulk_dir).")
    if not motifs_dir:
        raise ValueError("motifs_dir is required (pass --motifs_dir).")
    stage, tissue, cell_type, num_cells = get_file_name_parameters(schic_dataset_path)
    if not stage:
        stage = 'brain'
    
    cell_type = cell_type.replace(' ', '_')
    
    rna_seq_file = os.path.join(rna_seq_dataset_path, 'chr{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
    schic_file =  os.path.join(schic_dataset_path, 'chr{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
    
    bulk_hic_file = os.path.join(bulk_hic_dir, 'chr{}_{}.npz'.format(chromosome, PARAMETERS['resolution']))
    informative_indexes_bulk_hic = bulk_hic_file

    ctcf_motif_file = os.path.join(motifs_dir, 'ctcf', 'chr{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
    cpg_motif_file = os.path.join(motifs_dir, 'cpg', 'chr{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
    dataset_labels = json.load(open(DATASET_LABELS_JSON, 'r'))
    
    border_size = PARAMETERS['remove_borders'] // PARAMETERS['resolution']
    
    # Bulk Hi-C (prior)
    bulk_hic_object = np.load(bulk_hic_file, allow_pickle=True)
    # Use compact indexes from the same bulk Hi-C file as the prior.
    informative_indexes = np.load(informative_indexes_bulk_hic, allow_pickle=True)['compact']
    
    bulk_hic_data = compactM(bulk_hic_object['hic'], informative_indexes)
    bulk_hic_data = bulk_hic_data[border_size:, border_size:]
    
    # Divide
    bulk_hic_data, _ = divide_matrix(bulk_hic_data, chromosome, PARAMETERS)
    
    if PARAMETERS['bulk_hic'] == 'basic_prior':
        replace_with_prior = lambda adj: generate_expected_contact_matrix(adj)
        bulk_hic_data = bulk_hic_data.reshape([bulk_hic_data.shape[0], -1])
        bulk_hic_data = np.apply_along_axis(replace_with_prior, 1, bulk_hic_data)
    
    # Features
    rna_seq_data = np.load(rna_seq_file)
    ctcf_motif_data = np.load(ctcf_motif_file)
    cpg_motif_data = np.load(cpg_motif_file)
    
    # Take informative indices only
    rna_seq_data = rna_seq_data.take(informative_indexes, axis=1)
    ctcf_motif_data = ctcf_motif_data.take(informative_indexes, axis=1)
    cpg_motif_data = cpg_motif_data.take(informative_indexes, axis=1)
    
    # Clip Borders
    rna_seq_data = rna_seq_data[:, border_size:]
    ctcf_motif_data = ctcf_motif_data[:, border_size:]
    cpg_motif_data = cpg_motif_data[:, border_size:]
    
    node_features = rna_seq_data
    if PARAMETERS['ctcf_motif'] == True:
        node_features = np.concatenate((node_features, ctcf_motif_data))
    
    if PARAMETERS['cpg_motif'] == True:
        node_features = np.concatenate((node_features, cpg_motif_data))

    node_features, _ = divide_signal(node_features.T, chromosome, PARAMETERS)    
    node_features = node_features[:, 0, :, :]

    # scHi-C data
    schic_data = np.load(schic_file)
    schic_data = compactM(schic_data, informative_indexes)
    schic_data = schic_data[border_size: , border_size:]
    
    # Divide
    schic_data, indexes = divide_matrix(schic_data, chromosome, PARAMETERS)
    
    
    
    schic_data = schic_data.reshape([schic_data.shape[0], -1])
    bulk_hic_data = bulk_hic_data.reshape([bulk_hic_data.shape[0], -1])
    
    if PARAMETERS['normalization_algorithm'] == 'library_size_normalization':
        normalization_function = lambda adj: normalizations[PARAMETERS['normalization_algorithm']](adj, PARAMETERS['library_size'])
        schic_data = np.apply_along_axis(normalization_function, 1, schic_data)
        bulk_hic_data = np.apply_along_axis(normalization_function, 1, bulk_hic_data)
    else:
        normalization_function = lambda adj: normalizations[PARAMETERS['normalization_algorithm']](adj)
        schic_data = np.apply_along_axis(normalization_function, 1, schic_data)
        bulk_hic_data = np.apply_along_axis(normalization_function, 1, bulk_hic_data)
    
    schic_data = schic_data.reshape(schic_data.shape[0], 1, schic_data.shape[1], schic_data.shape[2]) 
    bulk_hic_data = bulk_hic_data.reshape(bulk_hic_data.shape[0], 1, bulk_hic_data.shape[1], bulk_hic_data.shape[2]) 
    
    parameterized_graph_pe = lambda adj: graph_pe(adj, encoding_dim=PARAMETERS['pos_encodings_dim'])
    pe = bulk_hic_data.reshape([bulk_hic_data.shape[0], -1])
    pe = np.apply_along_axis(parameterized_graph_pe, 1, pe)
    
    if PARAMETERS['hic_smoothing']:
        smooth_parameterized = lambda adj: smooth_adjacency_matrix(adj, PARAMETERS['smoothing_threshold'])
        schic_data = schic_data.reshape([schic_data.shape[0], -1])
        schic_data = np.apply_along_axis(smooth_parameterized, 1, schic_data)
    
    metadata = np.array([
        [dataset_labels['stage'][stage]]*indexes.shape[0],
        [dataset_labels['tissue'][tissue]]*indexes.shape[0],
        [dataset_labels['cell_type'][cell_type]]*indexes.shape[0],
        [int(num_cells)]*indexes.shape[0]
    ]).T
    
    return node_features[4:-4, :, :], schic_data[4:-4, :, :, :], pe[4:-4, :, :], bulk_hic_data[4:-4, :, :, :], indexes[4:-4, :], metadata[4:-4, :]



def chromosome_dataset_mp(args):
    '''
        Stupid arg passing trick for multiprocessing
    '''
    return create_chromosome_dataset(*args)



def create_cell_type_dataset(rnaseq_folder, schic_folder, PARAMETERS, set='debug', bulk_hic_dir=None, motifs_dir=None):
    chromosomes = dataset_partitions[set]

    # Creating arg lists for the multiprocessor
    chromosomes = list(map(lambda x: str(x), chromosomes))
    args = zip(
        repeat(rnaseq_folder),
        repeat(schic_folder),
        chromosomes,
        repeat(PARAMETERS),
        repeat(bulk_hic_dir),
        repeat(motifs_dir)
    )
    
    num_cpus = multiprocessing.cpu_count() if MULTIPROCESSING else 1
    if num_cpus >= len(chromosomes):
        num_cpus = len(chromosomes)
    
    
    with Pool(num_cpus) as pool:
        results = pool.map(chromosome_dataset_mp, args)
    

    results = list(filter(lambda x: len(x) != 0, results))

    node_features = np.concatenate([r[0] for r in results])
    targets = np.concatenate([r[1] for r in results])
    pes = np.concatenate([r[2] for r in results])
    bulk_hic = np.concatenate([r[3] for r in results])
    indexes = np.concatenate([r[4] for r in results])
    metadatas = np.concatenate([r[5] for r in results])
    
    return node_features, targets, pes, bulk_hic, indexes, metadatas



def create_schic_pseudobulk_dataset(exclusion_set, PARAMETERS, set='debug', descriptor='pb',
                                    rnaseq_input_dir=None, schic_input_dir=None,
                                    bulk_hic_dir=None, motifs_dir=None, output_dir=None):
    if not bulk_hic_dir:
        raise ValueError("bulk_hic_dir is required (pass --bulk_dir).")
    if not rnaseq_input_dir:
        raise ValueError("rnaseq_input_dir is required (pass --rnaseq_dir).")
    if not schic_input_dir:
        raise ValueError("schic_input_dir is required (pass --schic_dir).")
    if not output_dir:
        raise ValueError("output_dir is required (pass --output_dir).")
    if not motifs_dir:
        raise ValueError("motifs_dir is required (pass --motifs_dir).")
    create_directory(output_dir)

    scrnaseq_dataset_files = list(map(
        lambda x: os.path.join(rnaseq_input_dir, x),  # previously MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCRNASEQ
        os.listdir(rnaseq_input_dir))
    )
    scrnaseq_dataset_files = list(filter(
        lambda x: '.csv' not in x,
        scrnaseq_dataset_files
    ))
    
    schic_dataset_paths = []
    scrnaseq_dataset_paths = []
    for scrnaseq_dataset_file in scrnaseq_dataset_files:
        # Check if the dataset has enough cells? 
        stage, tissue, cell_type, num_cells = get_file_name_parameters(scrnaseq_dataset_file)
        
        # Update the json dictionary
        add_dataset(stage, tissue, cell_type)
        
        
        # Exclusion criterion
        if num_cells < PARAMETERS['num_cells_cutoff']:
            continue
        
        if tissue in exclusion_set or stage in exclusion_set or cell_type in exclusion_set:
            continue
        
        folder = '_'.join([stage, tissue, cell_type, 'n{}'.format(num_cells), 'schic']) if stage else  '_'.join([tissue, cell_type, 'n{}'.format(num_cells), 'schic'])
        
        schic_folder_path = os.path.join(schic_input_dir, folder)  # was MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCHIC
                
        if os.path.exists(schic_folder_path):
            schic_dataset_paths.append(schic_folder_path)
            scrnaseq_dataset_paths.append(scrnaseq_dataset_file)
    
    nfs = []
    tars = []
    pes = []
    bhs = []
    idxes = []
    metadatas = []
    
    output_file = os.path.join(
        output_dir,  # was MOUSE_PROCESSED_DATA_HIRES
        '{}_{}.npz'.format(descriptor, set)
    )
    
    for rnaseq_folder, schic_folder in zip(scrnaseq_dataset_paths, schic_dataset_paths):
        print('Working with: ', rnaseq_folder, ' and ', schic_folder)
        
        nf, tar, pe, bh, idx, meta = create_cell_type_dataset(
            rnaseq_folder,
            schic_folder,
            PARAMETERS,
            set,
            bulk_hic_dir=bulk_hic_dir,
            motifs_dir=motifs_dir,
        )
        nfs.append(nf)
        tars.append(tar)
        pes.append(pe)
        bhs.append(bh)
        idxes.append(idx)
        metadatas.append(meta)
    
    
    
    nfs = np.concatenate(nfs)
    tars = np.concatenate(tars)
    pes = np.concatenate(pes)
    bhs = np.concatenate(bhs)
    idxes = np.concatenate(idxes) 
    metadatas = np.concatenate(metadatas)
    
    print(nfs.shape, tars.shape, pes.shape, bhs.shape, idxes.shape, metadatas.shape)
    
    print('Saving file:', output_file)
    
    
    np.savez_compressed(output_file, 
        node_features=nfs, 
        targets=tars,
        pes=pes,
        bulk_hics=bhs,
        indexes=idxes,
        metadatas=metadatas
    )


# Standalone CLI for building the .npz dataset from parsed pseudobulk matrices.
if __name__ == '__main__':
    import argparse
    import src.globals as globals_module
    import src.utils as utils_module

    p = argparse.ArgumentParser(description='Build the .npz dataset from parsed pseudobulk matrices.')
    p.add_argument('--rnaseq_dir', required=True, help='Directory of parsed pseudobulk scRNA-seq track folders.')
    p.add_argument('--schic_dir', required=True, help='Directory of parsed pseudobulk scHi-C matrix folders.')
    p.add_argument('--bulk_dir', required=True, help='Directory of the bulk Hi-C prior (.npz per chromosome).')
    p.add_argument('--motifs_dir', required=True, help='Directory with ctcf/ and cpg/ motif tracks.')
    p.add_argument('--output_dir', required=True, help='Directory to write {experiment}_{set}.npz.')
    p.add_argument('--dataset_labels', default=None, help='dataset_labels.json (defaults to globals).')
    p.add_argument('--set', default='train', help='Chromosome partition: train/valid/test/ood/debug.')
    p.add_argument('--experiment', default='scgraphic', help='Output file name prefix ({experiment}_{set}.npz).')
    p.add_argument('--exclusion', nargs='*', default=[], help='stage/tissue/cell_type values to exclude.')
    p.add_argument('--no_multiprocessing', action='store_true', help='Disable multiprocessing (run single-process).')

    # Processing parameters (defaults mirror initialize_parameters_from_args)
    p.add_argument('--resolution', type=int, default=50000)
    p.add_argument('--pos_encodings_dim', type=int, default=16)
    p.add_argument('--normalization_algorithm', default='library_size_normalization')
    p.add_argument('--num_cells_cutoff', type=int, default=190)
    p.add_argument('--bulk_hic', default='mesc')
    p.add_argument('--library_size', type=float, default=25000)
    p.add_argument('--remove_borders', type=int, default=30000000)
    p.add_argument('--smoothing_threshold', type=float, default=0.25)
    p.add_argument('--stride', type=int, default=32)
    p.add_argument('--num_nodes', type=int, default=128)
    p.add_argument('--bounds', type=int, default=10)
    p.add_argument('--hic_smoothing', action='store_true', default=True)
    p.add_argument('--no_hic_smoothing', dest='hic_smoothing', action='store_false')
    p.add_argument('--padding', action='store_true', default=True)
    p.add_argument('--no_padding', dest='padding', action='store_false')
    p.add_argument('--ctcf_motif', action='store_true', default=True)
    p.add_argument('--no_ctcf_motif', dest='ctcf_motif', action='store_false')
    p.add_argument('--cpg_motif', action='store_true', default=True)
    p.add_argument('--no_cpg_motif', dest='cpg_motif', action='store_false')

    args = p.parse_args()

    if args.no_multiprocessing:
        globals()['MULTIPROCESSING'] = False

    if args.dataset_labels:
        globals()['DATASET_LABELS_JSON'] = args.dataset_labels
        globals_module.DATASET_LABELS_JSON = args.dataset_labels
        utils_module.DATASET_LABELS_JSON = args.dataset_labels

    PARAMETERS = {
        'resolution': args.resolution,
        'pos_encodings_dim': args.pos_encodings_dim,
        'normalization_algorithm': args.normalization_algorithm,
        'num_cells_cutoff': args.num_cells_cutoff,
        'bulk_hic': args.bulk_hic,
        'library_size': args.library_size,
        'remove_borders': args.remove_borders,
        'hic_smoothing': args.hic_smoothing,
        'smoothing_threshold': args.smoothing_threshold,
        'ctcf_motif': args.ctcf_motif,
        'cpg_motif': args.cpg_motif,
        'stride': args.stride,
        'num_nodes': args.num_nodes,
        'padding': args.padding,
        'bounds': args.bounds,
    }

    create_schic_pseudobulk_dataset(
        args.exclusion, PARAMETERS,
        set=args.set, descriptor=args.experiment,
        rnaseq_input_dir=args.rnaseq_dir,
        schic_input_dir=args.schic_dir,
        bulk_hic_dir=args.bulk_dir,
        motifs_dir=args.motifs_dir,
        output_dir=args.output_dir,
    )    
            
        
    