import os
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from clip import helpers

def morgan_from_smiles(smiles, radius=3, nbits=1024, chiral=True):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nbits, useChirality=chiral)
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp,arr)
    return arr


if __name__ == '__main__':
    indir = "<path-to-your-folder>"
    index = "<path-to-your-folder>/cellpainting-index.csv"
    index = os.path.join(indir, index)

    outdir = "/publicwork/sanchez/data/"
    outfile_hdf = "morgan_chiral_fps.hdf5"
    outfile_hdf = os.path.join(outdir, outfile_hdf)

    n_cpus = 60

    csv = pd.read_csv(index)

    csv["ID"] = csv.apply(lambda row: "-".join([str(row["PLATE_ID"]), str(row["WELL_POSITION"]),  str(row["SITE"])]), axis=1)

    ids = csv["ID"]
    smiles = csv["SMILES"]

    fps = helpers.parallelize(morgan_from_smiles, smiles, n_cpus)

    columns = [str(i) for i in range(fps[0].shape[0])]

    df = pd.DataFrame(fps, index=ids, columns=columns)
    df.to_hdf(outfile_hdf, key="df", mode="w")

