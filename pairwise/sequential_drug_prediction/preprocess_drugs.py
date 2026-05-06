import os
import gzip
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def preprocess_drugs(format='maccs', file_path=None, pert_ids=None):
    morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
    records = []
    opener = gzip.open if file_path.endswith('.gz') else open
    with opener(file_path, 'rt', encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        col = {h.strip(): i for i, h in enumerate(header)}
        
        pert_ids_lower = {p.lower() for p in pert_ids} if pert_ids is not None else None
        for line in f:
            fields = line.rstrip('\n').split('\t')
            pert_id = fields[col['pert_id']].strip()
            pert_iname = fields[col['pert_iname']].strip()
            smiles = fields[col['canonical_smiles']].strip()

            matched = (
                pert_ids_lower is None
                or pert_id.lower() in pert_ids_lower
                or pert_iname.lower() in pert_ids_lower
            )
            if matched and smiles not in ('', '-666', 'restricted'):
                m = Chem.MolFromSmiles(smiles)
                if m is not None:
                    fp = None
                    if format == 'maccs':
                        fp = np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.float32)
                    elif format == 'morgan':
                        fp = np.array(morgan_gen.GetFingerprintAsNumPy(m), dtype=np.float32)
                    records.append({
                        'pert_id': pert_id,
                        'pert_iname': pert_iname,
                        'smiles': smiles,
                        'fp': fp
                    })

    df = pd.DataFrame(records).drop_duplicates(subset=['pert_id', 'pert_iname']).reset_index(drop=True)
    print(f'Found {len(df)} unique compounds')
    
    return df