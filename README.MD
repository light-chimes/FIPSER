# The Implementation of FIPSER

### The Feature Importance and Perturbation potential-based SEed prioRitization method (FIPSER), is a seed selection method used for individual discrimination instance search algorithms.

This repository provides the code implementation for FIPSER. This implementation is based on the code
of [DICE](https://github.com/armanunix/Fairness-testing)
Consequently, the structure of this repository is similar to the structure
of [DICE](https://github.com/armanunix/Fairness-testing), which includes:

- [DICE_baseline](./NF_baseline) includes the implementation of baseline NeuronFair
- [DICE_data](./DICE_data) includes functions to load datasets
- [DICE_model](./DICE_model) includes the implementation of DNN model
- [DICE_tutorial](./DICE_tutorial) includes the source code of ADF, EIDIG and DICE
- [DICE_utils](./DICE_utils) includes utility functions of DICE
- [Raw_data](./Raw_data/raw) includes raw dataset files(meps15/16)
- [Clusters](./clusters) includes pickles from KMeans clustering
- [Datasets](./datasets) includes pre-processed datasets
- [Models](./models) includes trained models checkpoints
- [Requirements](./Requirements.txt) set of required libraries for running the tool on local machine
- [License](./LICENSE) file
- [fairness_improvement](./fairness_improvement) includes the implementation of model retraining and evaluation

# Requirements

Python 3.8  
numpy==1.22.0  
pandas==1.5.1  
tensorflow==2.7.0  
scipy==1.4.1  
argparse==1.1  
protobuf==3.9.2  
scikit-learn==0.22.2.post1   
aif360==0.4.0  
IPython==7.13.0  
regex==2022.10.31

# Usage

For the experiment of effectiveness and efficiency, use DICE_tutorial/ADF.py, DICE_tutorial/EIDIG.py,
DICE_tutorial/DICE_Search.py, and NF_baseline/NF_main/src/NF.py

For the experiment of retraining, use fairness_improvement/data_process.py to split datasets,
fairness_improvement/pretrain.py to pretrain models, fairness_improvement/retrain.py to retrain models, and
fairness_improvement/eval.py to evaluate the accuract and fairness of models before or after retraining.

Before running the fairness_improvement/retrain.py, put the search result of DICE into
fairness_improvement/generated_idip, rename it to \[dataset\]\_\[index of sensitive attribute (starting from 1)\]\_\[sampling method\]here's an example:

> generated_idip

>> heart_2_cluster

>>>2_10runs

>>>> global_inputs_0.npy

> > > > global_inputs_90_0.csv

> > > > local_inputs_0.npy

> > > > local_inputs_90_0.csv

> > > > QID_RQ2_10runs.npy

> > > > RQ2_table.csv

> > > > total_disc_0.csv

> > > > total_inputs_0.npy