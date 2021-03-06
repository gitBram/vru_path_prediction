from helpers import path
import sys, os, yaml
import numpy as np
import matplotlib.pyplot as plt

from helpers.waypoint_analyser import WaypointAnalyser
from helpers.highlevel_sceneloader import HighLevelSceneLoader
from helpers.graph import Graph


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

    # grid_x_min = int(min(scene_data.traj_dataframe['pos_x']) - margin) # [m]
    # grid_x_max = int(max(scene_data.traj_dataframe['pos_x']) + margin) # [m]
    # grid_y_min = int(min(scene_data.traj_dataframe['pos_y']) - margin) # [m]
    # grid_y_max = int(max(scene_data.traj_dataframe['pos_y']) + margin)  # [m]

    grid_x_min = 4
    grid_x_max = 110
    grid_y_min = -75
    grid_y_max = -2

    grid_limits = (grid_x_min, grid_x_max, grid_y_min, grid_y_max)

    
    # WPA = WaypointAnalyser(scene_data.traj_dataframe, grid_res, grid_limits, look_around_dist)
    # interest_areas = WPA.interest_area_searcher(savepath = 'data/images/WPA/areas.png')
    # interest_points = WPA.interest_point_searcher(interest_areas, savepath = 'data/images/WPA/points.png', min_dist = 3)
    
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
    n, l = g.analyse_full_signal(df_signals[60], False)

    # create figure and axes
    fig1, ax1 = plt.subplots()
    scene_data.plot_on_image([df_signals[60], g.points_locations, l], 
    save_path='data/images/example_paths/example_path.png', ms = [3, 6, 6])

    fig2, ax2 = plt.subplots()
    scene_data.plot_on_image([df_signals[60], g.points_locations, l], ms = [3, 6, 6])

    scene_data.add_circles(g.points_locations, 4, save_path='data/images/example_paths/example_path_c.png', ax=ax1)

    ''' do some probability predictions '''

    '''
    # recalculate the matrices
    g.recalculate_trans_mat_dependencies()
    
   
    # for analysis in prob_checker.ipynb
    import pickle
    with open('data/pickle/mat.pickle', 'wb') as handle:
        pickle.dump(g.trans_mats_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/pickle/dict.pickle', 'wb') as handle:
        pickle.dump(g.points_indices_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    g.calculate_destination_probs(['d7'])
    '''

    # displaying the probabilities of the destinations 
    g.recalculate_trans_mat_dependencies()
    dest_probs_dict = g.calculate_destination_probs(n)
    dest_probs_l = list(dest_probs_dict.values())
    dest_locs_dict_l = g.destination_locations    
    scene_data.plot_dest_probs(dest_locs_dict_l, dest_probs_l, 3, 200, ax1, 'data/images/dest_pred/dest_pred.png')
    
    # displaying the probabilities of all the waypoints and destinations 
    dest_probs_dict = g.calculate_destination_probs(n, "points")
    dest_probs_l = list(dest_probs_dict.values())
    dest_locs_dict_l = g.points_locations    
    scene_data.plot_dest_probs(dest_locs_dict_l, dest_probs_l, 3, 200, ax1, 'data/images/dest_pred/points_pred.png')
    

    # let's have a look what the probabilitites look like when we walk across the street
    n2 = ['w9', 'wa', 'w3', 'w6']
    dest_probs_dict = g.calculate_destination_probs(n2)
    dest_probs_l = list(dest_probs_dict.values())
    dest_locs_dict_l = g.destination_locations    
    scene_data.plot_dest_probs(dest_locs_dict_l, dest_probs_l, 3, 200, ax2, 'data/images/dest_pred/dest_pred2.png')
    
    # and cross again!
    n3 = ['w9', 'wa', 'w3', 'w6', 'wb']
    dest_probs_dict = g.calculate_destination_probs(n3)
    dest_probs_l = list(dest_probs_dict.values())
    dest_locs_dict_l = g.destination_locations    
    scene_data.plot_dest_probs(dest_locs_dict_l, dest_probs_l, 3, 200, ax1, 'data/images/dest_pred/dest_pred3.png')
    
    print(g.num_dest_arrivers_dict)

    s_ind = g.points_indices_dict['d7']
    d_ind = g.points_indices_dict['d6']
    for i in range(1, len(g.trans_mats_dict.keys())):
        tm = g.trans_mats_dict[i]
        print('Num of steps: {:.0f}, Prob: {:.4f}'.format(i, tm[s_ind, d_ind]))
        print(tm[s_ind, d_ind])

    # print(g.return_n_most_likely_dests(31, n))
    print(g.return_n_most_likely_next_points([45,-32], 3))

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