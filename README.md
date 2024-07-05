# Deep Sketched Output Kernel Regression for Structured Prediction

This Python package contains code to use kernel-induced losses with neural networks for Structured Prediction.

## Environment Preparation
```bash
conda create -n dsokr python=3.8
conda activate dsokr
conda install pytorch==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.37.2 \
            grakel==0.1.10 \
            torchtext==0.17.0 \
            scikit-learn==1.3.0 \
            tqdm==4.65.0 \
            numpy==1.24.3 \
            matplotlib==3.7.2 \
            rdkit==2023.9.4 \
            chainer-chemistry==0.7.1
```

## Experiment: SMI2Mol
### 1) Dataset
In order to create the dataset SMI2Mol used in our paper, please run the following script:
```bash
python create_smi2mol.py
```
The resulting data files can be found in the `Data/smi2mol` directory.

### 2) Sketching size selection with *Perfect h*
In order to select the sketching size with the *Perfect h* strategy described in our paper, you can run the following line:
```bash
python s2m_dsokr_perfect_h.py --random_seed_split 1996
```
When the experiment ends, you obtain the following figure, which provides a clue about the sketching size to choose:



### 3) Training and testing
Run the following script to train the `dsokr` model and get the performance on the test set:
```bash
python s2m_dsokr.py --random_seed_split 1996 --output_kernel "CORE-WL" --mys_kernel 3200 --nlayers 6 --nhead 8 --dropout 0.2 --dim 256
```
Normally, you can get the results like this (the results and the checkpoints are saved in the directory `exper`):
```
Test mean edit distance: 2.9375
Test mean edit distance w/o edge feature: 1.9905
```

## Experiment: SMI2Mol
