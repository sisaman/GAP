# Requirements
This code has been tested with [Python 3.9](https://www.python.org/downloads/) and requires the following packages.

To replicate the results of the paper please follow the below steps:

Step 1:
-------
Run nb.experiments.ipynb notebook. It creates a file "jobs/gap.jobs" containing all individual commands for running the experiments in the paper with different combinations of hyperparameters. You must specify your WandB username and (optional) project name in this notebook.

Step 2:
-------
execute the following command to run all the methods from "jobs/gap.jobs" one by one:
$ python jobs.py -f jobs/gap.jobs exec --all

This will train all the methods and log the results in the WandB project you specified in Step 1. 
WARNING: This step will take a lot of time. For faster execution, consider running the commands in the "jobs/gap.jobs" file in parallel or using a distributed job scheduler.

Step 3:
-------
Run nb.results.ipynb notebook to visualize the results as shown in the paper. It will fetch the experiment results from the WandB server, so you shoud set the same WandB username and project as in Step 1 in this notebook as well.
