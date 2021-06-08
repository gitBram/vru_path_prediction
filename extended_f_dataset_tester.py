from helpers import path
import sys, os, yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from helpers.waypoint_analyser import WaypointAnalyser
from helpers.highlevel_sceneloader import HighLevelSceneLoader
from helpers.graph import Graph
from predictors.dataset_creator import TFDataSet

def main():
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
    scene_data.load_ind(root_datasets, 11)

    ''' analyse data in order to get the waypoints '''
    grid_res = 1
    margin = 2
    look_around_dist = 3

    interest_points = __return_waypoints_ind()

    scene_data.plot_on_image([interest_points], save_path='data/images/data_vis/waypoints.png', col_num_dicts=dict(zip(["x", "y"], [0, 1])))
    scene_data.plot_dests_on_img('data/images/data_vis/destinations.png', col_num_dicts=dict(zip(["x", "y"], [0, 1])))

    ''' Fit a graph to the found points '''
    print(scene_data.destination_matrix)
    g = Graph.from_matrices(interest_points, scene_data.destination_matrix, 4, .05)

    df_signals = scene_data.df_to_lst_realxy_mats()
    g.analyse_multiple_full_signals(df_signals, add_to_trams_mat=True)
    print(g.trans_mat)

    my_graph = g.create_graph(.05)
    g.visualize_graph(my_graph, 'data/images/graphs/graph_with_image.png', scene_loader = scene_data)

    ''' test of displaying a path and its triggered waypoints '''
    extra_features_dict = {
        "all_points": None,
        "all_destinations": None,
        "n_destinations": 5,
        "n_points": 5
    }
    my_ds = TFDataSet.init_as_fixed_length(scene_data.traj_dataframe.head(200), scale_list=["pos_x", "pos_y"], seq_in_length=5, label_length=1, seq_stride=1,
    extra_features_dict=extra_features_dict, graph=g)

    # save the dataset
    # with open('data/pickle/dataset.pickle', 'wb') as f:
    #     pickle.dump(my_ds, f)

    normed, denormed = my_ds.example_dict("train", "in_xy")
    my_in, my_out = denormed
        # PLOT BASIC PREDICTION
    fig1, ax1 = plt.subplots()
    
    lbls = my_in["labels"][0].to_tensor()
    scene_data.plot_on_image([my_in["in_xy"][0], lbls], 
    save_path='data/images/extra_f/in_out.png', ms = [6, 1], ax=ax1,
    col_num_dicts=[my_ds.generalised_in_dict, my_ds.generalised_out_dict])

    dest_l = my_in["all_points"][0,:,0:2]
    dest_p = tf.squeeze(my_in["all_points"][0,:,2])

    scene_data.plot_dest_probs(dest_l, dest_p, 3, 200, ax = ax1, save_path = 'data/images/extra_f/destination_probs.png')
    
    a = np.ones((2,5,2))
    b = np.ones((5,2))
    aa = 2*a
    bb = 2*b
    print(scene_data.return_accuracy(a, aa))
    print(scene_data.return_accuracy(b, bb))
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
    [ 80, -52],
    [ 68, -58],
    [ 65, -54],
    [ 48, -20],
    [ 64, -21],
    [ 46, -14]])

    return d

if __name__ == '__main__':
    main()