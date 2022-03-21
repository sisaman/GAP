# GAP: Differentially Private Graph Neural Networks with Aggregation Perturbation

This repository is the official implementation of the paper:  
[**GAP: Differentially Private Graph Neural Networks with Aggregation Perturbation**](https://arxiv.org/abs/2203.00949)

## Results
<img src="https://i.imgur.com/Xlv0E7E.png" alt="results" width="500"/>

## Requirements

This code is implemented in Python 3.9, and requires the following modules:  
- [PyTorch](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [Opacus](https://opacus.ai/)
- [Autodp](https://github.com/yuxiangw/autodp)
- [WandB](https://docs.wandb.com/)
- [OGB](https://ogb.stanford.edu/docs/home/)
- [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/pages/quickstart.html)
- [Rich](https://rich.readthedocs.io/en/stable/introduction.html)
- [Ninja](https://ninja-build.org/)
- [Tabulate](https://github.com/astanin/python-tabulate)
- [Seaborn](https://seaborn.pydata.org/)
- [Jupyter](https://jupyter.org/install)
- [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/user_install.html)

Refer to [requiresments.txt](./requirements.txt) for the tested versions of the above packages.

### Notes
1. The code includes a custome C++ operator or faster edge sampling required for the node-level DP methods. PyTorch will automatically build the C++ code at runtime, but you need to have a C++ compiler installed (usually it is handled automatically if you use conda).

2. We use [Weights & Biases](https://docs.wandb.ai/) (WandB) to track the training progress and log experiment results. To replicate the results of the paper as described in the following, you need to have a WandB account. Otherwise, if you just want to train and evaluate the model, a WandB account is not required.

3. The code requires autodp version 0.2.1b or later. You can install the latest version directly from the [GitHub repository](https://github.com/yuxiangw/autodp) using:
```
pip install git+https://github.com/yuxiangw/autodp
```


## Usage

### Replicating the paper's results
To reproduce the paper's results, please follow the below steps:  

1. Run [experiments.ipynb](./experiments.ipynb) notebook. It creates a file "jobs/experiments.sh" containing all individual commands for running the experiments in the paper. You must specify your WandB username and (optionally) project name in this notebook.

2. Run ``sh jobs/experiments.sh`` to run all the experiments one by one. This will train all the methods and log the results to the WandB project you specified in step 1. 
WARNING: This step will take a lot of time. For faster execution, consider running the commands in the jobs in parallel or using a distributed job scheduler.

3. Run [results.ipynb](./results.ipynb) notebook to visualize the results as shown in the paper. It will fetch the experiment results from the WandB server, so you must set the same WandB username and project as in step 1 in this notebook as well. Note that we used the [Linux Libertine](https://libertine-fonts.org/) font in the figures, so you either need to have this font installed or change the font in the notebook.

### Training individual models

To train and evaluate the GAP model, run 
```
python src/train.py gap --help
``` 
to see the list of available options. Analogously, you can run 
```
python src/train.py sage --help
``` 
for the list of parameters of the GraphSAGE model.


## Contact

If you run into any problems or had any questions, please contact the author at: [sina.sajadmanesh@epfl.ch](mailto:sina.sajadmanesh@epfl.ch) or open an issue on [GitHub](https://github.com/sisaman/GAP/issues).


## Citation

If you find this code useful, please cite the following paper:  
```bibtex
@article{sajadmanesh2022gap,
  title={GAP: Differentially Private Graph Neural Networks with Aggregation Perturbation},
  author={Sajadmanesh, Sina and Shamsabadi, Ali Shahin and Bellet, Aur{\'e}lien and Gatica-Perez, Daniel},
  journal={arXiv preprint arXiv:2203.00949},
  year={2022}
}
```
