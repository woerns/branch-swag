# Branch-SWAG

This repository contains the implementation of Branch Stochastic Weight Averaging-Gaussian (Branch-SWAG) algorithm 
and the experimental framework to produce the results presented in [_Towards Scalable Bayesian Deep Learning with Sampled 
Branching Networks_](https://github.com/woerns/branch-swag/blob/main/branch-swag.pdf).

Branch-SWAG builds on [SWAG](https://arxiv.org/abs/1902.02476) which has been shown to be a simple and effective way to 
generate samples for Bayesian model averaging of neural networks and achieving state-of-the-art performance. Instead of 
sampling an entire network as proposed in SWAG, Branch-SWAG applies a sub-sampling technique that generates a stochastic
branching network which is more efficient in terms of computation and memory requirements without compromising its 
state-of-the-art performance. In fact, the experimental results demonstrate that sub-sampling can even improve 
performance in terms of test loss and calibration.


## Experiments

To run a cross-validation experiment enter the `code` directory and execute 
`run_experiment.py` with optional arguments. Here is an example:

    python run_experiment -r my_experiment -m densenet161 --swag --swag_branchout --dataset cifar-10 --device cuda --seed 42

If it runs successfully for the first time, the code will create the following folders in the root of this repo:

`datasets`: Folder with downloaded datasets from PyTorch.  
`logs`: Folder containing the log file for each experiment (e.g. `my_experiment.log`)  
`models`: Folder storing the training checkpoints and model configurations.  
`runs`: Folder containing TensorBoard log files. Run `tensorboard --logdir ./runs` from the repo 
root to view them in TensorBoard. 


## Parameters
Below is a list of main parameters. For a complete list of parameters please see `run_experiment.py`.

`--run_name`: Name of experiment. Will be automatically generated based on some parameter settings if not specified.

`--model_name`: Name of neural network architecture (e.g. `resnet50`). Currently only supports ResNet and DenseNet architectures implemented in PyTorch.

`--swag`: Flag to activate SWAG during training and evaluation.

`--swag_branchout`: Flag to evaluate model at all specified branchout layers.

`--swag_branchout_layers`: List of branchout layers. If not provided predefined list for each network architecture will be used.

`--dataset`: Name of dataset. Currently supported datasets are `cifar-10` and `cifar-100`.

`--device`: Either `cpu` or `cuda` to run on CPU or GPU, respectively.

`--seed`: List of random seeds to run.

## References

For the original SWAG implementation, please checkout [https://github.com/wjmaddox/swa_gaussian](https://github.com/wjmaddox/swa_gaussian)
