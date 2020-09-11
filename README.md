---   
<div align="center">    
 
# Computational insights into human perceptual expertise for familiar and unfamiliar face recognition    

[![Paper](http://img.shields.io/badge/Cognition-2020-4b44ce.svg)](https://psyarxiv.com/bv5mp)
<!--
PSYARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://psyarxiv.com/bv5mp)
-->
</div>
 
## Description   
- Deep convolutional neural network simulations of human unfamiliar and familiar face verification performance.
- Demonstration of dependence of human unfamiliar face recognition performance on learned face-specific statistics
- Account of the familiarity benefit in terms of a learned readout from fixed (or relatively stable) perceptual representations

### Citation   
```
@article{Blauch2020,
  title={Computational insights into human perceptual expertise for familiar and unfamiliar face recognition},
  author={Nicholas M. Blauch, Marlene Behrmann, David C. Plaut},
  journal={Cognition},
  year={2020}
}
```   

## Setup 

- Acquire **conda** 
- setup the main and secondary **conda** environments: **familiarity**, and **menpofit**: 
```
conda env create -f envs/familiarity.yml # main environment for DCNN simulations 
conda activate familiarity
conda install cudatoolkit=10.1 -c pytorch # if on linux or otherwise needed
pip install git+https://github.com/iitzco/faced.git # face detector needed to setup databases

conda deactivate
conda env create -f envs/menpofit.yml # secondary environment for conceptual replication of Kramer et. al (2018) paper
```

- Decide whether you are going to use SLURM HPC or local computation. Other HPC systems are possible but will require manual configuration (which we will not assist with). It would be very very time-consuming to do all of the from-scratch trainng serially on a local machine using VGG-16, but this option is available if desired. In any case, at least one modern GPU is required. 
 - If using SLURM, edit the file ```run_slurm.sbatch``` to adjust for your particular setup. 

- Acquire ImageNet and VGGFace2

- Configure paths in `familiarity/config.py`


## Run models
```
conda activate familiarity
bash scripts/run_all_pretraining.sh # pre-train VGG-16 on faces or objects
bash scripts/run_all_finetuning.sh # run the fine-tuning/familiarization experiments

conda activate menpofit
bash scripts/kramer_conceptual_replication.py
```  

## Data on KiltHub
- Alternatively to running the models yourself, you can download all of the pre-trained and fine-tuned models along with relevant results necessary for all plots in the paper on Kilthub: https://doi.org/10.1184/R1/12275381 

## Plot the results
View the jupyter notebooks; we recommend using Jupyter Lab (which can be installed and launched with:)
``` 
conda install -c conda-forge jupyterlab
jupyter-lab
```
You must have your paths configured and results produced, either through running the experiments yourself, or downloading the KiltHub data and configuring `familiarity.config.DATA_DIR` to point to the directory holding its data. 

**Results 3.1-3.2** : ```notebooks/plot_kramer.ipynb```  
**Results 3.3**: ```notebooks/plot_training_and_roc.ipynb```  
**Results 3.4-3.5**: ```notebooks/plot_verification_exp1.ipynb```   
**Results 3.6**: ```notebooks/plot_dists.ipynb```  
**Results 3.7**: ```notebooks/plot_fbf.ipynb```   
**Results 3.8**: ```notebooks/plot_verification_exp2.ipynb```    
**Results 3.9-3.10** : ```notebooks/plot_human_dcnn_comparison.ipynb```    

## questions and bugs: 
create an issue. if a bug, please describe in detail what you have tried and any steps necessary to reproduce the issue. 
