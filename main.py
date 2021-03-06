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

def main():
    ''' set some parameters '''
    # Model parameters
    LSTM_LAYER_SIZE = 64
    DENSE_LAYER_SIZE = 128
    NUM_LSTM_LAYERS = 2
    NUM_DENSE_LAYERS = 2
    VARIABLE_INPUT_LENGTH = True

    # Dataset
    SEQ_IN_LEN = 3
    SEQ_OUT_LEN = 5
    NOISE_STD = .3
    N_REPEATS = 1

    BATCH_SIZE = 1
    LENGTH_STRIDE = 2


    # Training parameters
    MAX_EPOCHS = 100
    PATIENCE = 10

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
    scene_data.load_ind(root_datasets, 7)


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
        "n_points": 5
    }

    # Load data in order to not need to do calculations again
    with open("data/pickle/ds_creation_d/bs1.pickle", 'rb') as handle:
        my_ds_creation_dict = pickle.load(handle)

    my_ds = TFDataSet.init_as_fixed_length(scene_data.traj_dataframe, graph=g, var_in_len=VARIABLE_INPUT_LENGTH, length_stride=LENGTH_STRIDE,
    scale_list=["pos_x", "pos_y"], seq_in_length=SEQ_IN_LEN, label_length=SEQ_OUT_LEN,
    extra_features_dict=extra_features_dict, noise_std=NOISE_STD, 
    n_repeats=N_REPEATS, batch_size=BATCH_SIZE, ds_creation_dict=my_ds_creation_dict) #save_folder = "data/pickle/ds_creation_d/bs1.pickle"

    ''' time for some model training '''
    # BASIC TRAINER
    my_trainer = DLTrainer(max_epochs=MAX_EPOCHS, patience=PATIENCE)
    my_trainer.LSTM_one_shot_predictor_named_i(my_ds, LSTM_LAYER_SIZE, DENSE_LAYER_SIZE, 
    NUM_LSTM_LAYERS, NUM_DENSE_LAYERS, extra_features=["all_destinations"], var_time_len=VARIABLE_INPUT_LENGTH)

    save_path = "data/model_weights/checkpoints/cp5/cp5.ckpt"

    my_trainer.compile_and_fit(my_ds, 'data/model_weights/checkpoints/bin/test_cp1.ckpt', test_fit=True)
    my_trainer.load_weights(save_path)
    # try:
    #     # first do one epoch of training in order to initialize weights
    #     my_trainer.compile_and_fit(my_ds, 'data/model_weights/checkpoints/bin/test_cp1.ckpt', test_fit=True)
    #     my_trainer.load_weights(save_path)
    # except:
    #     my_trainer.compile_and_fit(my_ds, save_path)
        
    ''' time for some model predictions '''
    my_predictor = extended_predictor(g, my_trainer, 1)
    
    nxt_unsc, nxt_sc = my_ds.example_dict("test", "in_xy")
    print(nxt_unsc)
    
    # Let's extract just one path to make visualisation clearer
    unscaled_ex = dict(nxt_unsc[0]), nxt_unsc[1]
    scaled_ex = dict(nxt_sc[0]), nxt_sc[1]
    # let's get an example of length one
    for key in unscaled_ex[0].keys():
        unscaled_ex[0][key] = tf.expand_dims(nxt_unsc[0][key][0], axis=0)
    for key in scaled_ex[0].keys():
        scaled_ex[0][key] = tf.expand_dims(nxt_sc[0][key][0], axis=0)    


    # Basic prediction, but repeated (one at a time)
    #PROBLEM: input is scaled
    assembled_output1, destination_list1, dest_prob_dict = my_predictor.predict_to_destinations(unscaled_ex[0], 10, 1, VARIABLE_INPUT_LENGTH)
    assembled_output2, destination_list2, dest_prob_dict = my_predictor.predict_to_destinations(unscaled_ex[0], 20, 1, VARIABLE_INPUT_LENGTH)
    # Epistemic uncertainty prediction
    # YET TO COME
    
    for i in range(9):
        fig1, ax1 = plt.subplots()
        scene_data.plot_on_image([unscaled_ex[0]["input_labels"], unscaled_ex[0]["labels"], assembled_output1[i]], 
        save_path='data/images/final_notebook/example_prediction'+str(i)+'.png', ms = [6, 1, 6], ax=ax1,
        col_num_dicts=[my_ds.generalised_in_dict, my_ds.generalised_out_dict,my_ds.generalised_out_dict], axes_labels=["input", "output", "prediction"])
        
        dest_locs = destination_list1[i][:, :, 0:2]
        dest_probs = destination_list1[i][:, :, 2:3]
        
        scene_data.plot_dest_probs(dest_locs[0], dest_probs[0], 2, 200,
        ax = ax1)

    for i in range(9):
        fig1, ax1 = plt.subplots()
        scene_data.plot_on_image([unscaled_ex[0]["input_labels"], unscaled_ex[0]["labels"], assembled_output2[i]], 
        save_path='data/images/final_notebook/example_prediction_long'+str(i)+'.png', ms = [6, 1, 6], ax=ax1,
        col_num_dicts=[my_ds.generalised_in_dict, my_ds.generalised_out_dict,my_ds.generalised_out_dict], axes_labels=["input", "output", "prediction"])
        
        dest_locs = destination_list2[i][:, :, 0:2]
        dest_probs = destination_list2[i][:, :, 2:3]
        
        scene_data.plot_dest_probs(dest_locs[0], dest_probs[0], 2, 200,
        ax = ax1)


    # PLOT BASIC PREDICTION
    fig1, ax1 = plt.subplots()

    # create dicts for correctly displaying data    
    scene_data.plot_on_image([scaled_ex, output], 
    save_path='data/images/predictions/example_prediction.png', ms = [6, 1], ax=ax1,
    col_num_dicts=[my_ds.generalised_in_dict, my_ds.generalised_out_dict])

    # LOT BASIC PREDICTION REPEATED
    fig2, ax2 = plt.subplots()

    # create dicts for correctly displaying data    
    scene_data.plot_on_image([scaled_ex, output_r], 
    save_path='data/images/predictions/example_prediction_r.png', ms = [6, 1], ax=ax2,
    col_num_dicts=[my_ds.generalised_in_dict, my_ds.generalised_out_dict])

    # PLOT EPISTEMIC 
    fig3, ax3 = plt.subplots()

    # create dicts for correctly displaying data    
    scene_data.plot_on_image([scaled_ex, output_e], 
    save_path='data/images/predictions/example_prediction_e.png', ms = [6, 1], ax=ax3,
    col_num_dicts=[my_ds.generalised_in_dict, my_ds.generalised_out_dict])

    return None

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


if __name__ == '__main__':
    main()