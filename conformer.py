import os
import torch
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool
from joblib import Parallel, delayed


def generate_conformer(args, df):
    print(f'Generate Conformers for {len(df)} SMILES (It takes time)...')
    if not os.path.exists(os.path.join(args.data_path, 'molecules')):
        os.makedirs(os.path.join(args.data_path, 'molecules'))
    smiles = df['SMILES'].unique()
    smiles_id = dict(zip(smiles, list(range(len(smiles)))))
    num_cpu = args.n_jobs
    smiles_split = np.array_split(smiles, num_cpu)
    conf_dict, coord_dict = {}, {}
    result = Parallel(n_jobs=num_cpu)(
        delayed(EmbedMultiConfs)(args, smi, smiles_id)
        for smi in smiles_split)
    for i_dict in result:
        conf_dict.update(i_dict[0])
        coord_dict.update(i_dict[1])
    del result

    df['Conformer_path'] = df['SMILES'].map(lambda x: conf_dict[x] if x in conf_dict.keys() else None)
    df = df.dropna(subset=['Conformer_path'])
    df.to_csv(os.path.join(args.data_path, 'molecule_structure.csv'), index=None)
    with open(os.path.join(args.data_path, 'molecule_structure_coord_dict.pkl'), 'wb') as f:
        pickle.dump(coord_dict, f)
    f.close()
    print('Conformer Generation Done!')

    return coord_dict


def EmbedMultiConfs(args, smiles, smiles_id,
                    numConfs: int = 10, maxAttempts: int = 100,
                    optimized=True):
    add_dict, coord_dict = {}, {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=numConfs,
                                   useRandomCoords=False,
                                   maxAttempts=maxAttempts,
                                   enforceChirality=True,
                                   useExpTorsionAnglePrefs=True,
                                   useBasicKnowledge=True)
        if len(mol.GetConformers()) == 0:
            AllChem.EmbedMultipleConfs(mol, numConfs=numConfs,
                                       useRandomCoords=True,
                                       maxAttempts=maxAttempts,
                                       enforceChirality=True,
                                       useExpTorsionAnglePrefs=True,
                                       useBasicKnowledge=False)
        new_mols, coord_feat = [], []
        for conf in mol.GetConformers():
            new_mol = Chem.Mol(mol)
            new_mol.RemoveAllConformers()
            new_mol.AddConformer(conf)
            if optimized:
                AllChem.MMFFOptimizeMolecule(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            new_mols.append(new_mol)
            coord_feat.append(get_coord_feat(new_mol.GetConformers()[0], is_conf=True))
            assert coord_feat[-1].shape[0] == Chem.MolFromSmiles(smi).GetNumAtoms()
        if len(coord_feat) == numConfs:
            coord_dict[smi] = torch.concat(coord_feat, dim=1)
        elif numConfs > len(coord_feat) > 0:
            coord_feat = pad_conformer(coord_feat, numConfs)
            coord_dict[smi] = torch.concat(coord_feat, dim=1)
        else:
            # print(f'SMILES {smi} has empty coordinate features!!!')
            continue
        conf_path = os.path.join(args.data_path, 'molecules', f'{smiles_id[smi]}.sdf')
        save_sdf(new_mols, conf_path)
        add_dict[smi] = conf_path
    return [add_dict, coord_dict]


def pad_conformer(coord_feat, numConfs):
    num = numConfs - len(coord_feat)
    sample_id = [i % numConfs for i in range(num)]
    for i in sample_id:
        coord_feat.append(coord_feat[sample_id[i]])
    return coord_feat


def save_sdf(mols, conf_path):
    writer = Chem.SDWriter(conf_path)
    writer.SetProps(['ConfID'])
    for i, mol in enumerate(mols):
        mol.SetProp('ConfID', str(i))
        writer.write(mol)
    writer.close()


def get_coord_feat(mol, is_conf=False):
    """Generate the node coordinate feature for a molecular graph.
        """
    if not is_conf:
        mol = mol.GetConformer()
    pos_feat_conf = []

    for i in range(mol.GetNumAtoms()):
        pos_feat_conf.append(list(mol.GetAtomPosition(i)))

    return torch.FloatTensor(pos_feat_conf)
