import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def preprocess_drugs(format='maccs', file_path=None, pert_ids=None):
    morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
    records = []
    with open(file_path, 'rt', encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        col = {h.strip(): i for i, h in enumerate(header)}
        
        for line in f:
            fields = line.rstrip('\n').split('\t')
            pert_id = fields[col['pert_id']].strip()
            smiles = fields[col['canonical_smiles']].strip()
            
            if pert_id in pert_ids and smiles is not None and smiles not in ('-666', 'restricted'):
                m = Chem.MolFromSmiles(smiles)
                if m is not None:
                    fp = None
                    if format == 'maccs':
                        fp = np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.float32)
                    elif format == 'morgan':
                        fp = np.array(morgan_gen.GetFingerprintAsNumPy(m), dtype=np.float32)
                    records.append({
                            'pert_id': pert_id,
                            'smiles': smiles,
                            'fp': fp
                        })
    
    df = pd.DataFrame(records).drop_duplicates(subset='pert_id').reset_index(drop=True)
    print(f'Found {len(df)} unique compounds')
    
    return df