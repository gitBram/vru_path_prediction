# Creating of data set and training of a network with variable length
# In .py file because ipynb always crashes.... Damn!


''' imports '''
import os, sys
from helpers.highlevel_sceneloader import HighLevelSceneLoader
from predictors.dataset_creator import TFDataSet
import tensorflow as tf
from predictors.dl_trainer import DLTrainer 
from predictors.extended_predictor import extended_predictor 
import matplotlib.pyplot as plt
import pickle
from helpers.graph import Graph
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from helpers.accuracy_functions import *
import csv
import math
from tqdm import tqdm

''' WPs list '''
''' Graph WPs '''
def __return_waypoints_ind():
    d = np.array([
    [ 65, -36],
    [ 56, -21],
    [ 66, -46],
    [ 58, -48],
    [ 67, -29],
    [ 61, -16],
    [ 45, -32],
    [ 71, -43],
    # [ 80, -52],
    # [ 68, -58],
    [ 65, -54],
    [ 48, -20],
    [ 64, -21],
    [ 46, -14]])
    return d


''' set some parameters '''
# Model parameters
LSTM_LAYER_SIZE = 64
DENSE_LAYER_SIZE = 128
NUM_LSTM_LAYERS = 2
NUM_DENSE_LAYERS = 2
VARIABLE_INPUT_LENGTH = True

# Dataset
SEQ_IN_LEN = 3
SEQ_OUT_LEN = 8
NOISE_STD = .3
N_REPEATS = 1

BATCH_SIZE = 1
LENGTH_STRIDE = 2


# Training parameters
MAX_EPOCHS = 100
PATIENCE = 5

# For graph
GRAPH_DIST_THRESH = 4

''' get the data '''
ROOT = os.getcwd()

rel_p_img_b = 'helpers/analysed_vars_storage/img_bounds.xml'
rel_p_dests = 'helpers/analysed_vars_storage/destination_locations.xml'
p_img_bounds = os.path.join(ROOT, rel_p_img_b)
p_dest_locs = os.path.join(ROOT, rel_p_dests)

#TODO: older version of OpenTraj needed: "git checkout d249ba6951dd0f54b532fbe2ca6edc46b0d7093f"
opentraj_root = os.path.join(ROOT, 'OpenTraj')
root_datasets = os.path.join(ROOT, 'data/path_data')
sys.path.append(opentraj_root) # add package to pythonpath

scene_data = HighLevelSceneLoader(p_img_bounds, p_dest_locs)
scene_data.load_ind(root_datasets, 7, 17)


''' create the graph instance '''    
interest_points = __return_waypoints_ind()
g = Graph.from_matrices(interest_points, scene_data.destination_matrix, GRAPH_DIST_THRESH, .05)

df_signals = scene_data.df_to_lst_realxy_mats()
g.analyse_multiple_full_signals(df_signals, add_to_trams_mat=True)

''' time to create df datasets '''
extra_features_dict = {
    "all_points": None,
    "all_destinations": None,
    "n_destinations": 5,
    "n_points": 5,
    "n_connected_points_after" : 3
}

# Load data in order to not need to do calculations again
with open("data/pickle/ds_creation_d/ds_7to17_inputLabelsvar3_8_test3.pickle", 'rb') as handle: #"data/pickle/ds_creation_d/bs1.pickle"
    my_ds_creation_dict = pickle.load(handle)

my_ds = TFDataSet.init_as_fixed_length(scene_data.traj_dataframe.head(300), graph=g, var_in_len=VARIABLE_INPUT_LENGTH, length_stride=LENGTH_STRIDE,
scale_list=["pos_x", "pos_y"], seq_in_length=SEQ_IN_LEN, label_length=SEQ_OUT_LEN,
extra_features_dict=extra_features_dict, noise_std=NOISE_STD, 
n_repeats=N_REPEATS, batch_size=BATCH_SIZE, ds_creation_dict=my_ds_creation_dict) # , save_folder = "data/pickle/ds_creation_d/ds_7to17_inputLabelsvar3_8.pickle"

# train the networks on the available data
# included
n_in_future = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
n_in_future = [1, 5, 9]
n_in_future = [9]

percentage_of_path_avail = [.2, .3, .4, .5, .6, .7, .8, .9, 1.]
num_input_steps = [2,3,4,5] #needs new network trained every time
num_output_steps = [3] #needs new network trained every time
dests_included = [False, True]
n_dense_layers = [2]
n_lstm_layers = [2]
dense_layer_sizes = [32, 128]
lstm_layer_sizes = [32, 128]
# not included

n_conn_points_incl = [False, True]
all_points_incl = [False, True]

# set up the csv writer
csv_headers = ["n", "ADE", "FDE", "counter", "not_counter", "num_in_steps", "num_out_steps", "dests_included", "all_points_included", "n_conn_points_included", "num_lstm_layers", "num_dense_layers", "lstm_layer_size", "dense_layer_size", "var_in_len"]
csv_folder_path = "data/results/destination_pred"
filename = 'results_' + datetime.now().strftime("%d_%m_%Y__%H_%M") + ".csv"
full_path = os.path.join(csv_folder_path, filename)

with open(full_path, 'w', newline='\n') as csvfile:
    my_writer = csv.writer(csvfile, delimiter=',')
    my_writer.writerow(csv_headers)

for n_dense in n_dense_layers:  
    for n_lstm in n_lstm_layers:
        for dest_included in dests_included:
            for conn_point in n_conn_points_incl:   
                for all_points in all_points_incl:
                    for dense_layer_size in dense_layer_sizes:
                        for lstm_layer_size in lstm_layer_sizes:
                            ''' TRAIN THE MODEL '''
                            # Create the model instance
                            feat_list = []
                            if dest_included:
                                feat_list.append("all_destinations")
                            if conn_point:
                                feat_list.append("n_connected_points_after")
                            
                            my_trainer = DLTrainer(max_epochs=MAX_EPOCHS, patience=PATIENCE)
                            my_trainer.LSTM_one_shot_predictor_named_i(my_ds, LSTM_LAYER_SIZE, DENSE_LAYER_SIZE, 
                            n_lstm, n_dense, extra_features=feat_list, var_time_len=VARIABLE_INPUT_LENGTH, size_dict=my_ds.size_dict)

                            folder_path = "data/model_weights/checkpoints/cp_path_pred_kpi/%s" % (datetime.now().strftime("%d_%m_%Y__%H_%M"))
                            save_path = os.path.join(folder_path, "dest%sconnp%sallp%s.pickle" % (dest_included, conn_point, all_points))            
                            if not os.path.exists(folder_path):
                                os.mkdir(folder_path)
                            my_trainer.compile_and_fit2(my_ds.tf_ds_dict["train"], my_ds.tf_ds_dict["val"], save_path=save_path)
                            ''' PLOT A BATCH AND PREDICTIONS FOR THIS MODEL '''
                            nxt_unsc, nxt_sc = my_ds.example_dict("test", "in_xy")
                            batch_out = my_trainer.predict_repetitively_dict(nxt_unsc[0], False, 10, VARIABLE_INPUT_LENGTH)

                            for i in range(BATCH_SIZE):
                                fig1, ax1 = plt.subplots()
                                scene_data.plot_on_image([nxt_sc[0]["in_xy"][i], nxt_sc[0]["labels"][i], batch_out[i]], 
                                save_path='data/images/train_exploration/example_prediction'+str(i)+'.png', ms = [6, 1, 3], ax=ax1, colors=["green", "blue", "red"],
                                axes_labels=["input", "output", "prediction"])
                            ''' GET KPIs FOR THIS MODEL '''   
                            print("Starting kpi calculation...")    
                            for n in tqdm(n_in_future):       
                                ade, fde, c, nc = tf_ds_kpi(my_ds.tf_ds_dict["test"], "labels", my_trainer, n, 
                                VARIABLE_INPUT_LENGTH)
                                
                                # Write them away for later analysis
                                line = [n, ade, fde, c, nc, SEQ_IN_LEN, SEQ_OUT_LEN, dest_included, all_points, conn_point, n_lstm, n_dense, lstm_layer_size, dense_layer_size, VARIABLE_INPUT_LENGTH]

                                with open(full_path, 'a', newline='\n') as csvfile:
                                    my_writer = csv.writer(csvfile, delimiter=',')
                                    my_writer.writerow(line)
                                


