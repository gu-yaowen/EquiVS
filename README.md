# EquiVS
Codes and data for "Employing Molecular Conformations for Ligand-based Virtual Screening with Equivariant Graph Neural Network and Deep Multiple Instance Learning"

# Reference
If you make advantage of the EquiVS model and our benchmark dataset proposed in our paper, please consider citing the following in your manuscript:

```
Gu Y, Li J, Kang H, et al. Employing Molecular Conformations for Ligand-based Virtual Screening with Equivariant Graph Neural Network and Deep Multiple Instance Learning[J]. 2023.
```

# Benchmark Dataset
Download our released dataset files from google driver:
https://drive.google.com/file/d/1mGNzxDVeczQzsxTPxIezUQWhOF5KRGE9/view?usp=sharing
Then unzip the "data" folder to this repo. The files are as shown:

>''data'' \
>>''molecules'' \
  The molecular conformer SDF files.
>>''target'' \
  The sub-datasets (CSV files) for specific targets, columns include 'Activity type', 'Ligand name', 'SMILES', 'Conformer_path', etc.
>>''molecule_structure.csv'' \
  The mapping file for each molecule SMILES and its corresponding conformer path in 'data' folder.

# EquiVS model
![EquiVS](https://github.com/gu-yaowen/EquiVS/blob/master/model%20structure.png)
## Environment Requirement
* torch==1.8.0
* dgl==0.5.2

## Train or Eval EquiVS model
    sh bash.sh
    
    
    Main arguments:
        -da: B-dataset C-dataset F-dataset R-dataset
        -ag: Aggregation method for bag embedding [sum, mean, Linear, BiTrans]
        -nl: The number of HeteroGCN layer
        -tk: The topk similarities in heterogeneous network construction
        -k : The topk filtering in instance predictor
        -hf: The dimension of hidden feature
        -ep: The number of epoches
        -bs: Batch size
        -lr: Learning rate
        -dp: Dropout rate
    For more arguments, please see main.py
    
# Contact
We welcome you to contact us (email: gu.yaowen@imicams.ac.cn) for any questions and cooperations.
