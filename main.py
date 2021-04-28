from helpers import path
import sys, os




def main():
    ROOT = os.getcwd()
    OPENTRAJ_ROOT = os.path.join(ROOT, 'OpenTraj')
    ROOT_DATASETS = os.path.join(ROOT, 'data/path_data')
    sys.path.append(OPENTRAJ_ROOT) 
    print('nu')
    print(OPENTRAJ_ROOT)
    from OpenTraj.toolkit.loaders import loader_ind
    from OpenTraj.toolkit.loaders import loader_sdd

    # import ind data
    ind_root = os.path.join(ROOT_DATASETS, 'inD-dataset-v1.0/data')
    file_id = 11 # range(0, 33)
    ind_dataset = loader_ind.load_ind(os.path.join(ind_root, '%02d_tracks.csv' % file_id),
                            scene_id='1-%02d' %file_id, sampling_rate=36, use_kalman=False)
    df_ind_trajectories = ind_dataset.data

if __name__ == '__main__':
    main()