import os, sys
from helpers.highlevel_sceneloader import HighLevelSceneLoader
from predictors.dataset_creator import TFDataSet
import tensorflow as tf
from predictors.dl_trainer import DLTrainer 
import matplotlib.pyplot as plt


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

    ''' time to create df datasets '''
    tf.executing_eagerly()
    my_ds = TFDataSet.init_as_fixed_length(scene_data.traj_dataframe, scale_list=["pos_x", "pos_y"], 
    seq_in_length=6, label_length=1, seq_stride=1, noise_std = .3, n_repeats=5)

    print(my_ds.example("train"))


    ''' time for some model training '''
    # BASIC TRAINER
    my_trainer = DLTrainer(max_epochs=60, patience=10)
    my_trainer.LSTM_one_shot_predictor(my_ds, 64, 128, 2, 2)

    save_path = "data/model_weights/example_predictor.h5"
    try:
        my_trainer.load_weights(save_path)
    except:
        my_trainer.compile_and_fit(my_ds, save_path)

    # EPISTEMIC TRAINER
    my_trainer_epi = DLTrainer(max_epochs=60, patience=10)
    my_trainer_epi.LSTM_one_shot_predictor_epi(my_ds, .2, 64, 128, 2, 2)
    save_path = "data/model_weights/example_predictor_ale.h5"
    try:
        my_trainer_epi.load_weights(save_path)
    except:
        my_trainer_epi.compile_and_fit(my_ds, save_path)

    ''' time for some model predictions '''
    nxt_unsc, nxt_sc = my_ds.example("test")
    unscaled_ex = nxt_unsc[0][0]
    scaled_ex = nxt_sc[0][0]

    # Basic prediction
    _, output = my_trainer.predict(unscaled_ex, scale_input_tensor = False)

    # Basic prediction, but repeated (one at a time)
    output_r = my_trainer.predict_repetitively(unscaled_ex, scale_input_tensor = False, num_repetitions=6, fixed_len_input=True)

    # Epistemic uncertainty prediction
    _, output_e = my_trainer_epi.predict(unscaled_ex, scale_input_tensor = False, n_evaluations = 50)

    # PLOT BASIC PREDICTION
    fig1, ax1 = plt.subplots()

    # create dicts for correctly displaying data    
    scene_data.plot_on_image([scaled_ex, output], 
    save_path='data/images/predictions/example_prediction.png', ms = [6, 6], ax=ax1,
    col_num_dicts=[my_ds.generalised_in_dict, my_ds.generalised_out_dict])

    # LOT BASIC PREDICTION REPEATED
    fig2, ax2 = plt.subplots()

    # create dicts for correctly displaying data    
    scene_data.plot_on_image([scaled_ex, output_r], 
    save_path='data/images/predictions/example_prediction_r.png', ms = [6, 6], ax=ax2,
    col_num_dicts=[my_ds.generalised_in_dict, my_ds.generalised_out_dict])

    # PLOT EPISTEMIC 
    fig3, ax3 = plt.subplots()

    # create dicts for correctly displaying data    
    scene_data.plot_on_image([scaled_ex, output_e], 
    save_path='data/images/predictions/example_prediction_e.png', ms = [6, 1], ax=ax3,
    col_num_dicts=[my_ds.generalised_in_dict, my_ds.generalised_out_dict])

    return None


if __name__ == '__main__':
    main()