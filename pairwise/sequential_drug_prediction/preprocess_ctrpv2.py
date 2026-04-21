import pandas as pd
import numpy as np
import pickle
import zipfile
import io
import argparse
from cmapPy.pandasGEXpress import parse_gctx
from preprocess_drugs import preprocess_drugs
from preprocess_lincs import ensure_decompressed
from train_val_test_split import compute_cell_line_baselines


def _read_zip_tsv(zf, name):
    with zf.open(name) as f:
        return pd.read_csv(io.TextIOWrapper(f), sep='\t', low_memory=False)


def preprocess_ctrpv2(args):
    """
    Generates (X, Y) for a cell viability regression model:
      X = delta gene expression (LINCS treatment - plate-matched DMSO control)
      Y = cpd_avg_pv (CTRP cell viability fraction)

    LINCS and CTRP samples are linked via shared broad_cpd_id and nearest
    log10 concentration (within max_log10_conc_delta).
    """
    print('Loading CTRPv2 data...')
    with zipfile.ZipFile(args.ctrpv2_zip) as zf:
        ctrp_raw = _read_zip_tsv(zf, 'v20.data.per_cpd_post_qc.txt')
        cell_info = _read_zip_tsv(zf, 'v20.meta.per_cell_line.txt')
        compound_info = _read_zip_tsv(zf, 'v20.meta.per_compound.txt')
        exp_info = _read_zip_tsv(zf, 'v20.meta.per_experiment.txt')

    # Join CTRP metadata: ccl_name and broad_cpd_id per viability row
    exp_info = exp_info.drop_duplicates(['experiment_id', 'master_ccl_id'])
    exp_info = exp_info.set_index('experiment_id')
    cell_info = cell_info.set_index('master_ccl_id')
    compound_info = compound_info.set_index('master_cpd_id')

    ctrp = ctrp_raw[['experiment_id', 'master_cpd_id', 'cpd_conc_umol', 'cpd_avg_pv']].copy()
    ctrp['master_ccl_id'] = exp_info.loc[ctrp['experiment_id'].values, 'master_ccl_id'].values
    ctrp['ccl_name'] = cell_info.loc[ctrp['master_ccl_id'].values, 'ccl_name'].values
    ctrp['broad_cpd_id'] = compound_info.loc[ctrp['master_cpd_id'].values, 'broad_cpd_id'].values
    ctrp = ctrp[['ccl_name', 'broad_cpd_id', 'cpd_conc_umol', 'cpd_avg_pv']]
    ctrp = ctrp.groupby(['ccl_name', 'broad_cpd_id', 'cpd_conc_umol']).mean().reset_index()
    ctrp['log10_conc'] = np.log10(ctrp['cpd_conc_umol'])
    print(f'CTRP: {len(ctrp)} rows after averaging replicates')

    print('Loading LINCS metadata...')
    inst_info = pd.read_table(
        ensure_decompressed(args.inst_info), sep='\t', header=0, low_memory=False
    )
    ctrp_cells = set(ctrp['ccl_name'])
    ctrp_compounds = set(ctrp['broad_cpd_id'])
    mask = (
        inst_info['cell_id'].isin(ctrp_cells) &
        inst_info['pert_id'].isin(ctrp_compounds) &
        (inst_info['pert_id'] != 'DMSO')
    )
    inst_lincs = inst_info[mask].copy()
    inst_lincs['pert_dose'] = pd.to_numeric(inst_lincs['pert_dose'], errors='coerce')
    inst_lincs = inst_lincs[inst_lincs['pert_dose'] > 0].copy()
    inst_lincs['log10_dose'] = np.log10(inst_lincs['pert_dose'])
    print(f'LINCS samples with CTRP match: {len(inst_lincs)}')

    ctrp_indexed = ctrp.set_index(['ccl_name', 'broad_cpd_id'])

    def _nearest_viability(row):
        key = (row['cell_id'], row['pert_id'])
        if key not in ctrp_indexed.index:
            return pd.Series({'cpd_avg_pv': np.nan, 'log10_ctrp_conc': np.nan})
        subset = ctrp_indexed.loc[[key]].copy()
        subset['delta'] = (subset['log10_conc'] - row['log10_dose']).abs()
        best = subset.nsmallest(1, 'delta').iloc[0]
        if best['delta'] > args.max_log10_conc_delta:
            return pd.Series({'cpd_avg_pv': np.nan, 'log10_ctrp_conc': np.nan})
        return pd.Series({'cpd_avg_pv': best['cpd_avg_pv'], 'log10_ctrp_conc': best['log10_conc']})

    print('Matching LINCS samples to nearest CTRP concentrations...')
    matched = inst_lincs.apply(_nearest_viability, axis=1)
    inst_lincs = pd.concat([inst_lincs, matched], axis=1)
    inst_lincs = inst_lincs.dropna(subset=['cpd_avg_pv']).reset_index(drop=True)
    print(f'{len(inst_lincs)} LINCS samples with a matched CTRP viability')

    print('Extracting LINCS expression profiles...')
    gctx_path = ensure_decompressed(args.train_path)
    landmark_genes_df = pd.read_table(
        ensure_decompressed(args.landmark_genes), sep='\t', header=0, low_memory=False
    )
    landmark_gene_ids = landmark_genes_df['pr_gene_id'].tolist()

    row_meta = parse_gctx.get_row_metadata(gctx_path)
    available_rids = set(row_meta.index.astype(str))
    landmark_gene_ids = [str(g) for g in landmark_gene_ids if str(g) in available_rids]

    all_dmso_ids = inst_info[inst_info['pert_id'] == 'DMSO']['inst_id'].tolist()
    all_ids = list(set(inst_lincs['inst_id'].tolist() + all_dmso_ids))
    gctx_data = parse_gctx.parse(gctx_path, cid=all_ids, rid=landmark_gene_ids)
    expr = gctx_data.data_df  # (genes, samples)

    print('Computing cell line consensus baselines from all DMSO replicates...')
    cell_line_baselines = compute_cell_line_baselines(inst_info, expr, landmark_gene_ids)

    inst_lincs = inst_lincs[inst_lincs['cell_id'].isin(cell_line_baselines)].reset_index(drop=True)

    trt_expr = expr[inst_lincs['inst_id'].tolist()].values.T  # (N, genes)
    ctl_expr = np.stack(
        [cell_line_baselines[cid] for cid in inst_lincs['cell_id']], axis=0
    )
    delta_expr = (trt_expr - ctl_expr).astype(np.float32)
    print(f'Delta expression shape: {delta_expr.shape}')

    print('Computing drug fingerprints...')
    drugs_df = preprocess_drugs(
        format=args.drug_format,
        file_path=ensure_decompressed(args.drug_info_path),
        pert_ids=set(inst_lincs['pert_id'].unique()),
    )
    drugs_df = drugs_df.set_index('pert_id')

    has_fp = inst_lincs['pert_id'].isin(drugs_df.index).values
    inst_lincs = inst_lincs[has_fp].reset_index(drop=True)
    delta_expr = delta_expr[has_fp]

    fp_dim = 167 if args.drug_format == 'maccs' else 1024
    n = len(inst_lincs)
    drug_fp = np.zeros((n, fp_dim), dtype=np.float32)
    for i, pid in enumerate(inst_lincs['pert_id']):
        drug_fp[i] = drugs_df.loc[pid, 'fp']

    with open(args.processed_data_dir, 'wb') as f:
        pickle.dump({
            'delta_expr': delta_expr,
            'drug_fp': drug_fp,
            'cell_ids': inst_lincs['cell_id'].values,
            'drug_names': inst_lincs['pert_iname'].values,
            'pert_time': inst_lincs['pert_time'].values,
            'concentration': inst_lincs['pert_dose'].values,
            'landmark_gene_ids': landmark_gene_ids,
            'cell_line_baselines': cell_line_baselines,
            'viability': inst_lincs['cpd_avg_pv'].values,
        }, f)
    print(f'Saved {n} samples to {args.processed_data_dir}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess CTRPv2 + LINCS for cell viability regression')
    parser.add_argument('--ctrpv2_zip', type=str, required=True,
                        help='Path to CTRPv2.0_2015_ctd2_ExpandedDataset.zip')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to LINCS Level3 gctx(.gz)')
    parser.add_argument('--inst_info', type=str, required=True,
                        help='Path to LINCS inst_info (.gz)')
    parser.add_argument('--drug_info_path', type=str, required=True,
                        help='Path to LINCS pert_info (.gz) for SMILES')
    parser.add_argument('--landmark_genes', type=str, required=True,
                        help='Path to landmark gene info file (.gz)')
    parser.add_argument('--drug_format', type=str, default='maccs', choices=['maccs', 'morgan'])
    parser.add_argument('--max_log10_conc_delta', type=float, default=0.2,
                        help='Max allowed log10 concentration difference for CTRP-LINCS matching')
    parser.add_argument('--processed_data_dir', type=str, required=True,
                        help='Output .pkl path')
    args = parser.parse_args()
    preprocess_ctrpv2(args)
