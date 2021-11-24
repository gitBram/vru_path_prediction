# Helpers
## Introduction

## Modules
### Accuracy Functions (accuracy_functions.py)
This module is used for calculation of KPIs identified to be useful for evaluating the models in the thesis. These include:
* Average displacement error;
* Final displacement error;
* Closest distance from trajectory to path.

### Graph (graph.py)
This module creates a one-step Markov Model (MM) which can be trained from a set of trajectories in a Pandas dataframe. For MM nodes, either a full grid or a selective grid (calculated by the waypoints analyzer) may be used. Training is done by snapping the trajectories to the grid and detecting emissions between MM nodes. Parameters for training can be set.
Apart from training functions, all functions for doing predictions based on an example path are also included in the module.

### Highlevel Sceneloader (highlevel_sceneloader.py)
The scene loader loads different data sets to a common format which can be interpreted by the other classes. Currently, the inD and SDD datasets are supported.
Also functions to show the dataset data visually and to plot trajectories on the scene background are implemented.

### Waypoints Analyzer (waypoint_analyzer.py)
This module contains a novel algorithm to extract waypoints and destination points based on a dataset with trajectories within a static scene. These are used for the creation of the MM.