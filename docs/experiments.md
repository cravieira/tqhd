# Experiment guide

This document describes how the experiments were executed and presents tips on reproducing them. It also explains some details behind the experimentation framework.

Each shell script named `paper-*.sh` is an *experiment* and outputs results to two possible folders: `_accuracy` and `_transformation`. The first folder contains the results for the floating-point HDC models, whereas the second stores the accuracies obtained by all different quantization techniques evaluated (TQHD, PQ-HDC, and QuantHD). There are also supplementary experiments that extend the TQHD's deviation experiment in the `script` folder. All scripts include the `script/common.sh` file defining common variables and routines.

Reproducing the experiments, as reported in the paper, may take several days. However, it is possible to split the execution into multiple machines and reduce the number of seeds evaluated (the paper evaluates 20 different seeds). For instance, setting `MAX_SEED=5` in `common.sh` greatly improves execution time.

## Experiment setup

The experiments for the paper were executed on two machines with CPU i9-12900KF, 32GB RAM, GPU RTX 3080, and Ubuntu 22.04. The number of parallel jobs in experiment scripts was tuned according to these machines. Consider adjusting the number of parallel jobs according to your available resources.

## Dynamic job selection

Finding the proper number of parallel jobs that your machine can execute in an experiment is an ad hoc process. Launching more jobs than your machine's RAM can handle will result in errors. On the other hand, underusing your machine's capabilities will result in a longer time to reproduce the results. Fortunately, there are some tricks that make the task of finding the proper number of jobs easier.

Each experiment calls the function `parallel_launch()` in `common.sh`. This function launches the commands necessary using GNU parallel and creates a temporary _procfile_. This file contains the number of jobs defined in the variable `JOBS` and can be tweaked to adjust the number of simultaneous jobs at runtime. I suggest starting with a low number of jobs and increasing them, always checking if your computer disposes enough free RAM before the increment. Each experiment script executes all applications (voicehd, language, mnist...), and some may require more resources than others. For instance, mnist usually requires more RAM to train than the other apps. Furthermore, some experiments vary the number of dimensions. These might require more RAM when working with bigger HDC models.

Please refer to GNU parallel documentation for more information related to procfiles.

## Running experiments in multiple machines

The scripting framework in this repository does not support any automatic way of splitting the experiments into multiple machines, which must be done manually by the user. There are two possible ways to achieve that:
1. Experiment dependency: The main `README.md` defines a dependency graph for the experiment scripts. The user can execute each part of the graph on a different machine.
2. Application dependency: Each experiment executes all applications. It is possible to execute a group of apps on a machine and the other apps on another computer. This is possible because the execution dependency between scripts arises only between the same app. However, doing this requires the user to edit the experiment files.

In both cases, the user must join the produced results to a single computer before generating plots with `paper-plots.sh`.

