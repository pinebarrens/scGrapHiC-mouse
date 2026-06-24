import enum
import os


import numpy as np
import pandas as pd



from src.globals import *
from src.utils import create_directory
from src.preprocess_datasets import read_cell_by_gene_matrix, read_pairix_file

def pseudobulk_rnaseq(tissue, cell_names, cell_type, umi_file, out_dir, stage=None):
    # umi_file = os.path.join(
    #     MOUSE_RAW_DATA_SCRNASEQ,
    #     'GSE223917_HiRES_{}.rna.umicount.tsv.gz'.format(tissue)
    # )
    cell_names_with_gene = ['gene'] + cell_names

    umi_data = read_cell_by_gene_matrix(umi_file)
    umi_data = umi_data.filter(cell_names_with_gene)
    umi_data[cell_type] = umi_data[cell_names].sum(axis=1)
    umi_data = umi_data.drop(cell_names, axis=1)

    umi_data[cell_type] = umi_data[cell_type]/len(cell_names)

    if stage:
        output_path = os.path.join(
            out_dir,  # was MOUSE_RAW_DATA_PSEUDO_BULK_SCRNASEQ
            '{}_{}_{}_n{}_umi.csv.gz'.format(stage, tissue, cell_type, len(cell_names))
        )

    else:
        output_path = os.path.join(
            out_dir,  # was MOUSE_RAW_DATA_PSEUDO_BULK_SCRNASEQ
            '{}_{}_n{}_umi.csv.gz'.format(tissue, cell_type, len(cell_names))
        )

    print('Saving pseudobulk UMI file at {}'.format(output_path))
    umi_data.to_csv(output_path, index=False)



def pseudobulk_schic(tissue, cell_names, cell_type, schic_file, out_dir, stage=None):
    # schic_files = list(map(
    #     lambda x: os.path.join(MOUSE_RAW_DATA_SCHIC, '{}.pairs.gz'.format(x)),
    #     cell_names
    # ))
    # pseudobulk_dataframe = read_pairix_file(schic_files[0])
    # for f in schic_files[1:]:
    #     pseudobulk_dataframe = pd.concat([pseudobulk_dataframe, read_pairix_file(f)])

    pseudobulk_dataframe = read_pairix_file(schic_file)
    pseudobulk_dataframe = pseudobulk_dataframe[pseudobulk_dataframe['readID'].isin(cell_names)]

    if stage:
        output_path = os.path.join(
            out_dir,  # was MOUSE_RAW_DATA_PSEUDO_BULK_SCHIC
            '{}_{}_{}_n{}_schic.pairs'.format(stage, tissue, cell_type, len(cell_names))
        )

    else:
        output_path = os.path.join(
            out_dir,  # was MOUSE_RAW_DATA_PSEUDO_BULK_SCHIC
            '{}_{}_n{}_schic.pairs'.format(tissue, cell_type, len(cell_names))
        )

    print('Saving pseudobulk scHi-C file at {}'.format(output_path))
    pseudobulk_dataframe.to_csv(output_path, index=False, header=False, sep ='\t')



def read_metadata_table(path):
    name = path.lower()
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    if name.endswith(('.tsv', '.tsv.gz', '.txt')):
        return pd.read_csv(path, sep='\t')
    if name.endswith(('.csv', '.csv.gz')):
        return pd.read_csv(path)

    return pd.read_csv(path, sep=None, engine='python')


def require_column(metadata, col, what):
    if col not in metadata.columns:
        raise KeyError(
            "{} column '{}' not found in metadata. Available columns: {}".format(
                what, col, list(metadata.columns)
            )
        )


def parse_metadata(metadata, celltype_col, barcode_col):
    require_column(metadata, celltype_col, 'Cell-type')
    require_column(metadata, barcode_col, 'Cell-barcode')

    cell_types = metadata[celltype_col].unique().tolist()
    cell_names = [
        metadata[metadata[celltype_col] == cell_type][barcode_col].tolist()
        for cell_type in cell_types
    ]
    return cell_types, cell_names


def create_pseudobulk_files(path, umi_file, schic_file, out_rnaseq, out_schic, tissue,
                            celltype_col='celltype', barcode_col='DNAbarcode', stage_col=None):
    create_directory(out_rnaseq)
    create_directory(out_schic)

    # metadata = pd.read_excel(path)
    metadata = read_metadata_table(path)

    # if tissue == 'embryo':
    #     # we repeat the process for all stages
    #     stages = metadata['Stage'].unique().tolist()
    #     for stage in stages:
    #         cell_types, cell_names = parse_metadata(metadata[(metadata['Stage'] == stage)])
    if stage_col:
        require_column(metadata, stage_col, 'Stage')
        for stage in metadata[stage_col].unique().tolist():
            subset = metadata[metadata[stage_col] == stage]
            cell_types, cell_names = parse_metadata(subset, celltype_col, barcode_col)
            for i, cell_type in enumerate(cell_types):
                pseudobulk_rnaseq(tissue, cell_names[i], cell_type, umi_file, out_rnaseq, stage)
                pseudobulk_schic(tissue, cell_names[i], cell_type, schic_file, out_schic, stage)
    else:
        cell_types, cell_names = parse_metadata(metadata, celltype_col, barcode_col)
        for i, cell_type in enumerate(cell_types):
            pseudobulk_rnaseq(tissue, cell_names[i], cell_type, umi_file, out_rnaseq, cell_type)
            pseudobulk_schic(tissue, cell_names[i], cell_type, schic_file, out_schic, cell_type)


# Standalone CLI for pseudobulk aggregation
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Pseudobulk raw scRNA-seq UMI and scHi-C .pairs by cell type.'
    )
    parser.add_argument('--metadata', required=True, help='Metadata table (.xlsx / .tsv / .csv).')
    parser.add_argument('--umi', required=True, help='Cell-by-gene UMI matrix (column 1 = gene).')
    parser.add_argument('--pairs', required=True, help='Merged single-cell .pairs file.')
    parser.add_argument('--out_rnaseq', required=True, help='Output dir for pseudobulk UMI .csv.gz files.')
    parser.add_argument('--out_schic', required=True, help='Output dir for pseudobulk .pairs files.')
    parser.add_argument('--tissue', default='brain')
    parser.add_argument('--celltype_col', default='celltype',
                        help='Metadata column holding the cell-type label.')
    parser.add_argument('--barcode_col', default='DNAbarcode',
                        help='Metadata column holding the cell barcode (matches .pairs readID and UMI columns).')
    parser.add_argument('--stage_col', default=None,
                        help='Optional metadata column to group by (e.g. developmental stage).')

    args = parser.parse_args()

    create_pseudobulk_files(
        args.metadata, args.umi, args.pairs,
        args.out_rnaseq, args.out_schic, args.tissue,
        celltype_col=args.celltype_col,
        barcode_col=args.barcode_col,
        stage_col=args.stage_col,
    )
