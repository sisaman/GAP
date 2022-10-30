# GAP: Differentially Private Graph Neural Networks with Aggregation Perturbation

This repository is the official implementation of the paper:  
[**GAP: Differentially Private Graph Neural Networks with Aggregation Perturbation**](https://arxiv.org/abs/2203.00949)

<!-- ## Results
<img src="https://i.imgur.com/Xlv0E7E.png" alt="results" width="500"/> -->

## Requirements

This code is implemented in Python 3.9, and requires the following modules:  
- [PyTorch](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [Opacus](https://opacus.ai/)
- [Autodp](https://github.com/yuxiangw/autodp)
- [WandB](https://docs.wandb.com/)
- [Dask-Jobqueue](https://jobqueue.dask.org/)
- [OGB](https://ogb.stanford.edu/docs/home/)
- [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/pages/quickstart.html)
- [Rich](https://rich.readthedocs.io/en/stable/introduction.html)
- [Ninja](https://ninja-build.org/)
- [Tabulate](https://github.com/astanin/python-tabulate)
- [Seaborn](https://seaborn.pydata.org/)
- [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/user_install.html)

Refer to [requiresments.txt](./requirements.txt) for the tested versions of the above packages.

## Notes
1. The code includes a custome C++ operator for faster edge sampling required for the node-level DP methods. PyTorch will automatically build the C++ code at runtime, but you need to have a C++ compiler installed (usually it is handled automatically if you use conda).

2. We use [Weights & Biases](https://docs.wandb.ai/) (WandB) to track the training progress and log experiment results. To replicate the results of the paper as described in the following, you need to have a WandB account. Otherwise, if you just want to train and evaluate the model, a WandB account is not required.

4. We use [Dask](https://jobqueue.dask.org/) to parallelize running multiple experiments on high-performance computing clusters (e.g., SGE, SLURM, etc). If you don't have access to a cluster, you can also simply run the experiments sequentially on your machine (see [usage section](#usage) below).

3. The code requires autodp version 0.2.1b or later. You can install the latest version directly from the [GitHub repository](https://github.com/yuxiangw/autodp) using: 
    ```
    pip install git+https://github.com/yuxiangw/autodp
    ```


## Usage

### Replicating the paper's results
To reproduce the paper's results, please follow the below steps:  

1. Set your WandB username in [wandb.yaml](./wandb.yaml) (line 7). This is required to log the results to your WandB account.

2. Execute the following python script:
    ```
    python experiments.py --generate
    ```
    This creates the file "jobs/gap.sh" containing the commands to run all the experiments.

3. If you want to run the experiments on your own machine, run:
    ```
    sh jobs/experiments.sh
    ``` 
    This trains all the models required for the experiments one by one. Otherwise, if you have access to a [supported HPC cluster](https://jobqueue.dask.org/en/latest/api.html), first configure your cluster setting (`~/.config/dask/jobqueue.yaml`) according to Dask-Jobqueue's [documentation](https://jobqueue.dask.org/en/latest/configuration.html). Then, run the following command:
    ```
    python experiments.py --run --scheduler <scheduler>
    ```
    where `<scheduler>` is the name of your scheduler (e.g., `sge`, `slurm`, etc). The above command will submit all the jobs to your cluster and run them in parallel. 
    

  4. Use [results.ipynb](./results.ipynb) notebook to visualize the results as shown in the paper. Note that we used the [Linux Libertine](https://libertine-fonts.org/) font in the figures, so you either need to have this font installed or change the font in the notebook.

### Training individual models

Run the following command to see the list of available options for training individual models:  

```
python train.py --help
``` 


## Contact

Should you ran into any problems or had any questions, please get in touch via [email](mailto:sina.sajadmanesh@epfl.ch) or open an issue on [GitHub](https://github.com/sisaman/GAP/issues).


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
