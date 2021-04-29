from helpers import path
import sys, os, yaml

from helpers.waypoint_analyser import WaypointAnalyser

def main():
    ''' get the data '''
    ROOT = os.getcwd()

    #TODO: older version of OpenTraj needed: "git checkout d249ba6951dd0f54b532fbe2ca6edc46b0d7093f"
    opentraj_root = os.path.join(ROOT, 'OpenTraj')
    root_datasets = os.path.join(ROOT, 'data/path_data')
    sys.path.append(opentraj_root) # add package to pythonpath

    # dataframe_ind = load_ind(root_datasets, 11)
    dataframe_sdd = load_sdd(opentraj_root, scene_name = 'gates', scene_video_id = 'video1')

    ''' analyse data in order to get the waypoints '''
    grid_res = 1
    margin = 2
    grid_x_min = int(min(dataframe_sdd['pos_x']) - margin) # [m]
    grid_x_max = int(max(dataframe_sdd['pos_x']) + margin) # [m]
    grid_y_min = int(min(dataframe_sdd['pos_y']) - margin) # [m]
    grid_y_max = int(max(dataframe_sdd['pos_y']) + margin)  # [m]
    grid_limits = (grid_x_min, grid_x_max, grid_y_min, grid_y_max)
    look_around_dist = 3
    WaypointAnalyser(dataframe_sdd, grid_res, grid_limits, look_around_dist)


def load_ind(root_datasets, file_id):

    from OpenTraj.toolkit.loaders import loader_ind

    # import ind data
    ind_root = os.path.join(root_datasets, 'inD-dataset-v1.0/data')
    ind_dataset = loader_ind.load_ind(os.path.join(ind_root, '%02d_tracks.csv' % file_id),
                            scene_id='1-%02d' %file_id, sampling_rate=36, use_kalman=False)
    return ind_dataset.data.reset_index()

def load_sdd(opentraj_root, scene_name, scene_video_id):
    from toolkit.loaders.loader_sdd import load_sdd, load_sdd_dir


    # fixme: replace OPENTRAJ_ROOT with the address to root folder of OpenTraj
    sdd_root = os.path.join(opentraj_root, 'datasets', 'SDD')
    annot_file = os.path.join(sdd_root, scene_name, scene_video_id, 'annotations.txt')
    image_file = os.path.join(sdd_root, scene_name, scene_video_id, 'reference.jpg')

    # load the homography values
    with open(os.path.join(sdd_root, 'estimated_scales.yaml'), 'r') as hf:
        scales_yaml_content = yaml.load(hf, Loader=yaml.FullLoader)
    scale = scales_yaml_content[scene_name][scene_video_id]['scale']

    ind_dataset = load_sdd(annot_file, scale=scale, scene_id=scene_name + '-' + scene_video_id,
                            drop_lost_frames=False,sampling_rate=36, use_kalman=False) 

    df_ind_trajectories = ind_dataset.data
    # only get pedestrians
    df_ind_trajectories=df_ind_trajectories[df_ind_trajectories['label']=='pedestrian']
    df_ind_trajectories = df_ind_trajectories.reset_index()

    return df_ind_trajectories

if __name__ == '__main__':
    main()