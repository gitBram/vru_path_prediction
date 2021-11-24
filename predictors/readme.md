# Predictors
## Introduction
This folder contains all scripts needed for the training of the RNN network within Tensorflow. 
## Modules
### Dataset Creator (dataset_creator.py)
This module converts a Pandas dataframe to an optimized set of Tensorflow datasets, being a dataset for training, testing and validating.
In general, the dataframe is expected to have two measures, being the x and y coordinates over time of a pedestrian. More dimensions may be added though. The column containing the pedestrian id also needs to be specified. Timestamps are not included as the time difference between steps is assumed to be constant and to be maintained during testing. 

For the resulting TF dataset, many options are available, including following.
* Shuffling data with specified shuffle buffer size.
* Enabling and disabling data normalization. Output predictions will be denormalized again if enabled.
* Splitting into train, test and validation datasets with specified percentages.
* Creating a fixed length or variable length (0 value padding) dataset.
* Batching in different sizes.
* Varying length of input and output trajectories in the dataset.
* Extra features can be added. Practically, this is used for inserting the destination estimates mainly as they get calculated before the creation of the Tensorflow dataset. 
### (De)Normalizer (de_normalizer.py)
Performance of neural networks is improved by normalizing the data. Based on a dataframe, the data is normalized for a selected set of dimensions. The normalization parameters for each dimension (standard deviation and average value) are stored to be used for denormalization of the network predictions. 

### Deep Learning Trainer (dl_trainer.py)
This module adds a higher level to training the RNN and doing predictions with it. This makes using the RNN created with **extended_predictor.py** easier to handle. Variables for the training may be set, such as number of epochs, patience, the loss function and metric to be used during training...

Functions are available to do prediction with extra features (such as destination prediction).

RNN networks with or without permanent dropout can be generated setting only simple variables like number of layers, layer sizes and dropout rate. 

### Extended Predictor (extended_predictor.py)
Functions are available to do prediction with extra features (such as destination prediction) and iteratred prediction for probabilistic outputs.