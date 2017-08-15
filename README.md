# CCSA: "Unified Deep Supervised Domain Adaptation and Generalization" (ICCV 2017)

## Requiremenrts
keras and numpy

## Introduction

This repository provides the implementation of the paper "Unified Deep Supervised Domain Adaptation and Generalization" published in ICCV 2017. It also contains the training/testing splits of two cross domain adaptation task (MNIST->USPS and USPS->MNIST). 

We are interested in the supervised domain adaptation when very few labeled target samples are available in training (from 1 to 7). 

Experimental setting involves randomly selecting 2000 images from MNIST and 1800 images from USPS. Here, we randomly selected n labeled samples per class from target domain data and used them in training. We evaluated our approach for n ranging from 1 to 7 and repeated each experiment 10 times. Therefore, we provided data we used to generate the results. Data files are located in the 'row_data' subdirectory.


### "We encourage researchers to use this data for comparison."



## Implementation

To reproduce the results of the paper you just need to run main.py. There are three main parameters:

1. sample_per_class = 1 or 2 or ... or 7 (sample_per_class specifies the number of labeled target data per class.)

2. repetition =  0 or 2 or ... or 9. (We repeat the experiments 10 times and report the average accuracies.)

3. domain_adaptation_task = 'MNIST_to_USPS' or 'USPS_to_MNIST'


There are some other hyperparameters that you may change for the new dataset.


## Citation

@InProceedings{motiian2017CCSA,
  Title                    = {Unified Deep Supervised Domain Adaptation and Generalization},

  Author                   = {Motiian, Saeid and Piccirilli, Marco and Adjeroh, Donald A. and Doretto, Gianfranco},

  Booktitle                = {IEEE International Conference on Computer Vision (ICCV)},

  Year                     = {2017}}
 
 
 
 
 For more information:
 
 http://vision.csee.wvu.edu/~motiian/Details/CCSA.html
 
 
 

