# Adversarial Ensemble Training by Jointly Learning Label Dependencies and Member Models

See the paper [here](https://arxiv.org/pdf/2206.14477.pdf).

This repository contains code for reproducing our conditional label dependency learning (CLDL) assisted ensemble training algorithm. In this code repository, we use 'LSD' to represent our CLDL based algorithm. We also include several other SOTA methods' codes for performance comparison.

## Empirical ensemble robustness

* ADP Training (https://arxiv.org/abs/1901.08846)

* GAL Training (https://arxiv.org/abs/1901.09981)

* DVERGE Training (https://arxiv.org/abs/2009.14720)

* Adversarial Training (https://arxiv.org/abs/1706.06083)

* TRS Training (https://arxiv.org/abs/2104.00671)

* **LSD Training** (ours)

## `eval` folder contains:
* `whitebox/blackbox.py`: Test the whitebox/blackbox attack robustness of the given ensemble model.

* `wbox_para_compare/bbox_para_compare.py`: Test the whitebox/blackbox attack robustness with different hyperparameter settings.

* `transferability.py`: Evaluate the adversarial transferability among base models under various attacks.

* `natural_acc.py`: Test the performance of the given ensemble model on normal examples.

* `surrogate.py`: Generate blackbox transfer attack instances from the given surrogate ensemble models. We choose three ensembles consisting of 3, 4 and 5 base models as our surrogate models. 
* `pick_out_correctly_idx.py`: Pick out the common test sub-dataset from entire test dataset on which all comparison ensemble models initially give correct classifications.

## `models` folder contains:
* `lcm.py`: Generate label confusion vector for members of the ensemble.


## `train` folder contains:

The corresponding code to construct above robust ensemble models. You can use the command as
`python train/train_xxx.py **kwargs`
`**kwargs` refers to the training parameters which is defined in `utils/arguments.py`

* `trainer.py`: the implementation of trainer of ensemble
```
    Baseline_Trainer: the implementation of trainer of Baseline 
    Adversarial_Trainer: the implementation of trainer of Adversarial 
    ADP_Trainer: the implementation of trainer of ADP
    DVERGE_Trainer: the implementation of trainer of DVERGE
    GAL_Trainer: the implementation of trainer of GAL
    LCM_GAL_Trainer: the implementation of trainer of our LSD
    TRS_Trainer: the implementation of trainer of TRS
```

## `utils` folder contains:


## Dependencies

We were using PyTorch 1.9.0 for all the experiments.


## Reference
If you find our paper/this repo useful for your research, please consider citing our work.
```

```

## Train LSD
`python train/train_lsd.py [options]`
 
Options:
* `--ld_coeff`: `coefficient for simulated_label_distribution`
* `--alpha`: `coefficient for jsd_loss`
* `--beta`: `coefficient for coherence`
* `--arch`: `model architecture`)
* `--model_num`: `number of submodels within the ensemble`)

```
python train/train_lsd.py --ld_coeff 4 --alpha 2 --beta 4 --seed 0 --model_num 3 --arch ResNet --dataset cifar10
```

Outputs:
Trained weights of each channel will be saved in `./checkpoints/lcm_gal_4.0_2.0_4.0/seed_0/3_ResNet20_cifar10/`.



## Test LSD

### 1、Pick out the common test sub-dataset from entire test dataset on which all comparison ensemble models initially give correct classifications.
`python eval/pick_out_correctly_idx.py`  

Outputs: 
results will be saved in `./data/pick_out_correctly_idx/`.


### 2、Generate blackbox transfer attack instances from the given surrogate ensemble models.
`python eval/surrogate.py`  

Outputs: 
results will be saved in `./data/transfer/`.


### 3、Test the whitebox/blackbox attack robustness with different hyperparameter settings.
`python eval/eval_wbox_para_compare.py`  
`python eval/eval_bbox_para_compare.py`  

Outputs: 
results will be saved in `./results/para_compare/0/`.


### 4、Test the performance of the given ensemble model on normal examples
`python eval/eval_natural_acc.py`  

Outputs: 
results will be saved in `./results/natural_acc/0/`.


### 5、evaluation of black-box robustness
`python eval/eval_bbox.py`  

Outputs: 
results will be saved in `./results/bbox/0/`.


### 6、evaluation of transferability
`python eval/eval_transferability.py`  

Outputs: 
results will be saved in `./results/transferability/0/`.


### 7、evaluation of white-box robustness
`python eval/eval_wbox.py`  

Outputs: 
results will be saved in `./results/wbox/0/`.


