# Can GNN be Good Adapter for LLMs?
This repository is mostly a duplication study for the implementation of GraphAdapter - [Can GNN be Good Adapter for LLMs?](https://arxiv.org/abs/2402.12984) in WWW 2024.

## Requirements
* python = 3.8
* numpy >= 1.19.5
* pytorch = 1 .10.2
* pyg = 2.3.1
* transformers >= 4.28.1

## Other requirements
Depending on the operating system, Torch version, and CUDA Driver version, the wheel may fail when updating and downloading default dependencies. To obtain them, you must find the exact specific specifications and download the exact package with pip using the URL for that file. 

Find these dependencies and binaries here: https://data.pyg.org/whl/torch-2.5.1%2Bcu124.html

This will need to be done for torch_sparse as well as some other packages as they're not immediately installed and the wheel often fails.

For the largest dataset Arxiv, 300G storage is required
## How to use our code
The datasets this paper used can be downloaded from [here](https://drive.google.com/drive/folders/13fqwSfY5utv8HibtEoLIAGk7k85W7b2d?usp=sharing), please download them and put them in datasets to unzip.


### Step 1. Preprocess data for training
```
python3 preprocess.py --dataset_name instagram --gpu 0 --plm_path llama2_path --type pretrain
```
The preprocess.py will load the textual data of Instagram, and next transform them to token embedding by Llama 2, which will be saved into saving_path. The saved embeddings will used in the training of GraphAdapter.

For our case, the plm_path argument will be meta-llama/Llama-2-13b-hf, however this assumes that huggingface has provided access to the closed/gated model.

### Step 2. Training GraphAdapter
```
python3 pretrain.py --dataset_name instagram --hiddensize_gnn 64 --hiddensize_fusion 64 --learning_ratio 5e-4 --batch_size 32 --max_epoch 15 --save_path your_model_save_path
```
An for the replication study, this worked:
```
python3 pretrain.py --dataset_name instagram --hiddensize_gnn 64 --hiddensize_fusion 64 --learning_ratio 5e-4 --gpu 0 --batch_size 32 --max_epoch 15
```
### Step 3. Finetuning for downstream task

GraphAdapter requires prompt embedding for finetuning,

```
python3 preprocess.py --dataset_name instagram --gpu 0 --plm_path llama2_path --type prompt

```
After preprocessing the dataset, now you can finetune to downstream tasks.
```
python3 finetune.py --dataset_name instagram  --gpu 0  --metric roc --save_path your_model_save_path 
```
Note: keep your_model_save_path consistent in both pretrain.py and finetune.py.

## Citation
If you find the original authprs' work or dataset useful, please consider citing their work:
```
@article{huang2024can,
  title={Can GNN be Good Adapter for LLMs?},
  author={Huang, Xuanwen and Han, Kaiqiao and Yang, Yang and Bao, Dezheng and Tao, Quanjin and Chai, Ziwei and Zhu, Qi},
  journal={WWW},
  year={2024}
}
```
