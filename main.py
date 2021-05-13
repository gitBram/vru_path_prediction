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

    grid_x_min = int(min(scene_data.traj_dataframe['pos_x']) - margin) # [m]
    grid_x_max = int(max(scene_data.traj_dataframe['pos_x']) + margin) # [m]
    grid_y_min = int(min(scene_data.traj_dataframe['pos_y']) - margin) # [m]
    grid_y_max = int(max(scene_data.traj_dataframe['pos_y']) + margin)  # [m]

    grid_x_min = 6
    grid_x_max = 110
    grid_y_min = -75
    grid_y_max = -2

    grid_limits = (grid_x_min, grid_x_max, grid_y_min, grid_y_max)

    
    WPA = WaypointAnalyser(scene_data.traj_dataframe, grid_res, grid_limits, look_around_dist)
    interest_areas = WPA.interest_area_searcher(save_img = True)
    interest_points = WPA.interest_point_searcher(interest_areas, save_img = True, min_dist = 3)
    
    # dest_points = np.array([
    #     [20,2],
    #     [30,2],
    #     [44,5],
    #     [45,22],
    #     [55,31],

    # ])
    scene_data.plot_on_image([interest_points], save_path='waypoints.png')
    scene_data.plot_dests_on_img('my_dests.png')

    ''' Fit a graph to the found points '''
    print(scene_data.destination_matrix)
    g = Graph.from_matrices(interest_points, scene_data.destination_matrix, 1., .2)
    my_graph = g.create_graph(.05)
    g.visualize_graph()


if __name__ == '__main__':
    main()