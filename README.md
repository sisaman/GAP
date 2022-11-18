# GAP: Differentially Private Graph Neural Networks with Aggregation Perturbation

This repository is the official implementation of the paper:  
[**GAP: Differentially Private Graph Neural Networks with Aggregation Perturbation**](https://arxiv.org/abs/2203.00949)   
Accepted at USENIX Security 2023


## Requirements

This code is implemented in Python 3.9 using PyTorch-Geometric 2.1.0 and PyTorch 1.12.1.
Refer to [requiresments.txt](./requirements.txt) to see the full list of dependencies.

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

## Results

<center>

| Privacy Level | Method        | $\epsilon$ | Facebook             | Reddit              | Amazon              |
|---------------|---------------|------------|----------------------|---------------------|---------------------|
| None          | GAP-$\infty$  | $\infty$   | 80.0 $\pm$ 0.48      | **99.4 $\pm$ 0.02** | 91.2 $\pm$ 0.07     |
| None          | SAGE-$\infty$ | $\infty$   | **83.2 $\pm$ 0.68**  | 99.1 $\pm$ 0.01     | **92.7 $\pm$ 0.09** |
| Edge    | GAP-EDP       | 4          | **76.3 $\pm$ 0.21**  | **98.7 $\pm$ 0.03** | **83.8 $\pm$ 0.26** |
| Edge    | SAGE-EDP      | 4          | 50.4 $\pm$ 0.69      | 84.6 $\pm$ 1.63     | 68.3 $\pm$ 0.99     |
| Edge    | MLP           | 0          | 50.8 $\pm$ 0.17      | 82.4 $\pm$ 0.10     | 71.1 $\pm$ 0.18     |
| Node    | GAP-NDP       | 8          | **63.2 $\pm$ 0.35**  | **94.0 $\pm$ 0.14** | **77.4 $\pm$ 0.07** |
| Node    | SAGE-NDP      | 8          | 37.2 $\pm$ 0.96      | 60.5 $\pm$ 1.10     | 27.5 $\pm$ 0.83     |
| Node    | MLP-DP        | 8          | 50.2 $\pm$ 0.25      | 81.5 $\pm$ 0.12     | 73.6 $\pm$ 0.05     |

</center>


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
