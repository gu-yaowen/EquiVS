from copyreg import pickle
import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from joblib import Parallel, delayed
from dgl import save_graphs, load_graphs
from dgllife.data import MoleculeCSVDataset
from dgl.data.utils import save_info, load_info
from dgllife.utils.splitters import RandomSplitter, ScaffoldSplitter
from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from conformer import generate_conformer


# fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
# chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


def load_dataset(args):
    # args.dataset = 'molecule_structure'
    mol_df = pd.read_csv(os.path.join(args.data_path, 'molecule_structure.csv'))
    del mol_df['Conformer_path']
    df = pd.read_csv(os.path.join(args.data_path, 'target', args.task + '.csv'))
    df = df.rename(columns={'Final Activity': 'Value'})
    # try:
    #     os.makedirs(args.data_save_path)
    # except:
    #     pass
    conf_path = os.path.join(args.data_path, 'molecules')
    graph_path = os.path.join(args.data_path, 'task',
                              args.task + '_processed',
                              'dgl_graph.bin')
    info_path = os.path.join(args.data_path, 'task',
                             args.task + '_processed',
                             'info.pkl')
    # del df['Conformer_path']
    if not os.path.exists(graph_path):
        if 'Conformer_path' not in mol_df.columns:
            if len(os.listdir(path=conf_path)) > 0:
                files = os.listdir(path=conf_path)
                for i in mol_df.index:
                    if str(i) + '.sdf' in files:
                        mol_df.loc[i, 'Conformer_path'] = str(i) + '.sdf'
                mol_df.to_csv(os.path.join(args.data_path, 'molecule_structure.csv'), index=None)
                coord_dict = {}
            else:
                coord_dict = generate_conformer(args, mol_df)
        elif 'molecule_structure_coord_dict.pkl' in os.listdir(path=os.path.join(args.data_path)):
            coord_dict = pickle.load(open(os.path.join(args.data_path,
                                                       'molecule_structure_coord_dict.pkl'), 'rb'))
        else:
            coord_dict = {}

        smiles_to_g = SMILESToBigraph(explicit_hydrogens=False, add_self_loop=True,
                                      node_featurizer=CanonicalAtomFeaturizer(atom_data_field='node_feat'),
                                      edge_featurizer=CanonicalBondFeaturizer(bond_data_field='edge_feat',
                                                                              self_loop=True))

        dataset = MoleculeCSVDataset(df=df,
                                     smiles_to_graph=smiles_to_g,
                                     smiles_column='SMILES',
                                     cache_file_path=graph_path,
                                     task_names='Value',
                                     init_mask=False,
                                     n_jobs=args.n_jobs)

        if coord_dict == {}:
            # have not been tested
            print('Get coordinate features from file...')
            len1 = len(df)
            df = df.dropna()
            len2 = len(df)
            print(f'{len1 - len2} samples are excluded')
            smiles_dict = dict(zip(list(range(len(dataset))), df['SMILES'].values.tolist()))
            split = np.array_split(list(range(len(dataset))), args.n_jobs)
            c_dict = Parallel(n_jobs=args.n_jobs)(
                delayed(get_coord)(args, idx_list, dataset, smiles_dict, df)
                for idx_list in split)
            for c in c_dict:
                coord_dict.update(c)
            del c_dict

        k = coord_dict.keys()
        filtered_graph, smiles, label, valid_idx = [], [], [], []
        generator = tqdm(range(len(dataset)), leave=False)
        for idx in generator:
            data = dataset[idx]
            generator.set_description("Processing %s" % idx)
            if data[0] in k:
                # print(Chem.MolFromSmiles(data[0]).GetNumAtoms())
                # print(data[1])
                data[1].ndata['coord_feat'] = coord_dict[data[0]]
                smiles.append(data[0])
                filtered_graph.append(data[1])
                label.append(data[-1])
                valid_idx.append(idx)

        print(f'{len(dataset) - len(valid_idx)} invalid datapoints have been filtered')
        print(f'{len(valid_idx)} valid datapoints have been assembled as dataset')
        save_graphs(graph_path, filtered_graph,
                    {'labels': torch.stack(label).float()})
        save_info(info_path, {'SMILES': smiles})
        filtered_graph = [(s, g, l) for s, g, l in zip(smiles, filtered_graph, label)]
        del coord_dict
    else:
        graphs, label_dict = load_graphs(graph_path)
        label = label_dict['labels'].float()
        smiles = load_info(info_path)['SMILES']
        filtered_graph = [(s, g, l) for s, g, l in zip(smiles, graphs, label)]

    return filtered_graph


def get_path(args, idx):
    return os.path.join(args.data_path, args.task, str(idx) + '.sdf')


def get_coord(args, idx_list, dataset, smiles_dict, df):
    path_list = os.listdir(path=os.path.join(args.data_path, 'molecules'))
    coord_dict = {}
    for idx in idx_list:
        path = get_path(args, idx)
        if path.split('/')[-1] not in path_list:
            print(f'SMILES {dataset[idx][0]} is not Found!')
            continue
        coord_feat = []
        try:
            mols = Chem.SDMolSupplier(path)
        except:
            print(f'SMILES {dataset[idx][0]} is not Found!')
            continue
        if len(mols) < 10:
            continue
        else:
            for i, mol in enumerate(mols):
                # mol = Chem.AddHs(mol)
                coord_feat.append(get_coord_feat(mol))
            coord_feat = torch.concat(coord_feat, dim=-1)
            coord_dict[idx] = coord_feat
    return coord_dict


def get_coord_feat(mol):
    """Generate the node coordinate features for a molecular graph.
        """
    conf = mol.GetConformer()
    pos_feat_conf = []

    for i in range(mol.GetNumAtoms()):
        pos_feat_conf.append(list(conf.GetAtomPosition(i)))

    return torch.FloatTensor(pos_feat_conf)


def dataset_split(dataset, args):
    # NOTE: Only random split with a specific ratio to train/val/test has been tested.
    args.k = 0
    args.idx_path = ''
    config = {'split': args.split,
              'random_state': args.seed,
              'k': args.k,
              'frac_train': args.frac_train,
              'frac_val': args.frac_val,
              'frac_test': args.frac_test,
              'scaffold_func': 'smiles',
              'idx_path': args.idx_path,
              'sanitize': True,
              'mols': None}
    if config['split'] == 'random':
        splitter = RandomSplitter()
        if config['k'] == 0:
            train, val, test = splitter.train_val_test_split(dataset,
                                                             frac_train=config['frac_train'],
                                                             frac_val=config['frac_val'],
                                                             frac_test=config['frac_test'],
                                                             random_state=config['random_state'])
            return train, val, test
        else:
            train, test = splitter.k_fold_split(dataset,
                                                k=config['k'],
                                                random_state=config['random_state'])
            val = None
    elif config['split'] == 'scaffold':
        splitter = ScaffoldSplitter()
        dataset = scaffold_dataset(dataset)

        if config['k'] == 0:
            train, val, test = splitter.train_val_test_split(dataset,
                                                             frac_train=config['frac_train'],
                                                             frac_val=config['frac_val'],
                                                             frac_test=config['frac_test'],
                                                             # random_state=config['random_state'],
                                                             scaffold_func='smiles',
                                                             sanitize=config['sanitize'],
                                                             mols=config['mols'])
        else:
            train, test = splitter.k_fold_split(dataset,
                                                k=config['k'],
                                                random_state=config['random_state'],
                                                scaffold_func='smiles',
                                                sanitize=config['sanitize'],
                                                mols=config['mols'])
            val = None
    elif config['split'] == 'customized':
        indices = pd.read_csv(config['idx_path'])
        tr, va, te = indices[indices['Dataset'] == 'train'], \
                     indices[indices['Dataset'] == 'val'], \
                     indices[indices['Dataset'] == 'test']
        tr_dict, va_dict, te_dict = dict(zip(tr.index.values, tr['SMILES'].values)), \
                                    dict(zip(va.index.values, va['SMILES'].values)), \
                                    dict(zip(te.index.values, te['SMILES'].values))
        smiles, idxs = indices['SMILES'].values, indices['index'].values

        train, val, test = [dataset[tr] for tr in tr_dict.keys() if dataset[tr].smiles == tr_dict[tr]], \
                           [dataset[va] for va in va_dict.keys() if dataset[va].smiles == va_dict[va]], \
                           [dataset[te] for te in te_dict.keys() if dataset[te].smiles == te_dict[te]]
    return train, val, test


class scaffold_dataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.smiles = [i[0] for i in dataset]

    def smiles(self, item):
        return self.smiles[item]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]