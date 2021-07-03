import numpy as np

class prediction_output:
    ''' Class for easily accessing a prediction result '''
    def __init__(self, predictions, input_traj, out_label_traj, dest_prob_dict, dest_loc_dict, 
        dest_index_dict, time_per_step, prediction_result):
        self._input_traj = np.array(input_traj)
        self._out_label_traj = np.array(out_label_traj)

        self._dest_prob_dict = dest_prob_dict
        self._dest_name_dict = dest_loc_dict
        self._dest_index_dict = dest_index_dict
        
        self._time_per_step = float(time_per_step)
        self._prediction_result = np.array(prediction_result)
        self._predictions = np.array(predictions)
        



    def get_prediction(self, weighted=False, dest=None, time=None):        

        return None

    @property
    def input_traj(self):
        return self._input_traj
    @input_traj.setter
    def input_traj(self, value):
        self._input_traj = value

    @property
    def dest_prob_dict(self):
        return self._dest_prob_dict
    @dest_prob_dict.setter
    def input_traj(self, value):
        self._dest_prob_dict = value

