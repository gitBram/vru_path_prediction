# Vulnerable Road User Location Prediction for Collision Avoidance Systems
This repo contains the implementation of the thesis "Vulnerable Road User Location Prediction for Collision Avoidance Systems".

## Proposal
The proposal can be found on Overleaf ([link](https://www.overleaf.com/read/ycmgwyfnywgd)).
## Code
Eventually, the project code will be maintained within this repository. For now, Google Colab is being used for technical reasons.
* Generation of simulated spline tracks ([link](https://colab.research.google.com/drive/1gqHE5SCddgbEs8Gs0xrQH1kI7EtMLNDZ?usp=sharing))
    * Features:
        * Add x,y coordinate lists in order to create 2D spline paths.
        * Define how many spline paths to make based on each coordinate list.
        * Add waiting areas (max 1 per spline segment) where people might or might not have to wait with certain probability, for a duration drawn from specified distribution. This aims to simulate a traffic light controlled crossing.
        * Each path is generated with a certain base speed, drawn from a specified distribution.
        * Add Gaussian measurement noise to the data.
        * Data exported to Google Drive folder (logon needed) in Google Sheet format.
        
* Basic structure for generating TF data sets and first tests with (probabilistic) neural networks ([link](https://colab.research.google.com/drive/1DCvEI3dFbwpTTodAf09PFo3vKNNgwkqc?usp=sharing)) 
    * Class proposal for data
        * Generated based on Pandas data frame
        * Possibility to generate TF data set which can be used for training/testing neural networks in TF
        * Easy visualisation of data and model predictions for this data
    * Tests and visualisations for basic models 
        * Dense Model
        * Convolutional Model
        * RNN (non-stateful) model
    * Tests and visualisations for probabilistic models
        * Post-Prediction Distribution Estimation
            * Permanent dropout layer(s)
        * At Prediction Distribution Estimation
            * Altered Loss Function
        
## Known Problems
* Google Colab "data_loading.ipynb": 
    * No train-validation-test split done in data class for now, therefore unreliable results since tested and validated with training data
    * No normalisation of data