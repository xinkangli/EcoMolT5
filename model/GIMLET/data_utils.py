from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import torch

def get_morgan_fingerprint(smiles_list, radius=2, n_bits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # 若SMILES非法，用0向量代替
            fps.append(np.zeros((n_bits,), dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)  # ✅ 用这个！
        fps.append(arr)
    return torch.tensor(fps)
