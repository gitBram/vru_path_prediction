# Simple Path Generator
## Introduction
This package is meant to generate spline paths, incorporating functionality like adding random waiting areas, adding random Gaussian noise to the measurements and randomized direction diversions to keep all recorded paths slightly different. 
## Purpose
This package is written with the purpose of being able to generate custom paths and use cases. This way, the applicability of the developed algorithms on these example cases can be tested.
These generated paths are in no way used for training of the final networks for use on a real scenario.
## Output format
timestamp, pedestrian id, x-position, y-position
