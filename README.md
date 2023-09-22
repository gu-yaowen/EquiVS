# EquiVS
[![HitCount](https://hits.dwyl.com/gu-yaowen/EquiVS.svg?style=flat-square&show=unique)](http://hits.dwyl.com/gu-yaowen/EquiVS)

Codes and data for "Employing Molecular Conformations for Ligand-based Virtual Screening with Equivariant Graph Neural Network and Deep Multiple Instance Learning"

# Reference
If you make advantage of the EquiVS model and our benchmark dataset proposed in our paper, please consider citing the following in your manuscript:

```
Gu, Y.; Li, J.; Kang, H.; Zhang, B.; Zheng, S. Employing Molecular Conformations for Ligand-Based Virtual Screening with Equivariant Graph Neural Network and Deep Multiple Instance Learning. Molecules 2023, 28, 5982. https://doi.org/10.3390/molecules28165982
        
        
        
          
```

# Benchmark Dataset
Download our released dataset files from [Google Drive](https://drive.google.com/file/d/1mGNzxDVeczQzsxTPxIezUQWhOF5KRGE9/view?usp=sharing). Then unzip the ``data`` folder to this repo. The files are as shown:

>``data``
>> * ``molecules`` \
  The molecular conformer SDF files.
>> * ``target`` \
  The sub-datasets (CSV files) for specific targets, columns include 'Activity type', 'Ligand name', 'SMILES', 'Conformer_path', etc.
>> * ``molecule_structure.csv`` \
  The mapping file for each molecule SMILES and its corresponding conformer path in 'data' folder.

Our bioactivity prediction dataset is curated from "A consensus compound/bioactivity dataset for data-driven drug design and chemogenomics"(https://zenodo.org/record/6398019#.YwuT93ZByUk). The data preprocessing cases can be found in [data_preprocessing.ipynb](https://github.com/gu-yaowen/EquiVS/blob/main/data_preprocessing.ipynb).

# EquiVS model
![EquiVS](https://github.com/gu-yaowen/EquiVS/blob/main/model_structure.png)
## Environment Requirement
* torch==1.8.0
* dgl==0.5.2

## Train or evaluate EquiVS model
Simply reproduce our model:
```
sh bash.sh
```
Note that the ``TARGET_processed`` folder will be created in ``data/task/`` to save molecular graphs (``dgl_graph.bin``) and labels (``info.pkl``). \
OR:
```
python main.py -id {DEVICE_ID} -sp {SAVE_PATH} -se {SEED} -mo {MODE: train OR eval}
```
For more (general, training, and model) arguments, please see ``main.py``.

The EquiVS model will automatically train on all stored sub-dataset in ``data/target`` folder. The training results, models, and predictions will be stored in ``result/{SAVE_PATH}`` folder. 

## Optimal conformer discovery
Please refer the model interpretation case in [model_interpret.ipynb](https://github.com/gu-yaowen/EquiVS/blob/main/model_interpret.ipynb).

# Contact
We welcome you to contact us (email: yg3191@nyu.edu) for any questions and cooperations.
