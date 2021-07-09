import numpy as np
import tensorflow as tf
# import remove_padding_vals from helpers.accuracy_functions

class extended_predictor:
    ''' 
    Class to do predictions that incorporate destination prediction
    '''
    def __init__(self, graph, dl_trainer, time_per_step):
        self._graph = graph
        self._dl_trainer = dl_trainer
        self._time_per_step = time_per_step
    
    def predict_to_destinations(self, input_dict, num_steps, num_predictions, variable_len_input):
        '''
        Predict the output based on an input dict 
        num_steps steps in to the future
        each prediction done num_prediction times (only useful if permanent dropout in model)

        Predictions done for dict with BS = 1
        '''
        # Predict output probabilities of the input using the graph
        #   Make sure that the transition matrix is updated
        self.graph.recalculate_trans_mat_dependencies() 
        #   Calculate the output probabilities
        input_lbl = self.remove_padding_vals(input_dict["input_labels"])
        print(input_lbl)
        visited_nodes,_ = self.graph.analyse_full_signal(input_lbl, False, allow_out_of_threshold=True)
        dest_prob_dict = self.graph.calculate_destination_probs(visited_nodes,"destinations")

        # Predict num_steps towards each destination (It's easy: dl_trainer handles scaling etc)
        # get list of all destinations + locations

        stack_list = [] # used for keeping track of output towards each destination
        destination_list = [] # used for keeping track of which destination is selected
        for destination in self.graph.destination_names:
            # get one hot vector towards this destination
            dest_index = self.graph.destinations_indices_dict[destination]
            min_dest_index = min(self.graph.destinations_indices_dict.values())

            new_probs = tf.one_hot(
                [dest_index-min_dest_index], self.graph.num_destinations, on_value=None, off_value=None, axis=None, dtype=tf.float32, name=None
            )
            new_probs = tf.reshape(new_probs, [-1, 1])

            in_shape = input_dict["in_xy"].shape
            if len(in_shape)==2:
                batch_size = 1
            else: 
                batch_size = in_shape[-3]
            probs_lst = [new_probs for i in range(batch_size)]
            new_probs=tf.stack(probs_lst)

            # implement the one hot vector
            input_dict_c = dict(input_dict)
            dest_loc_one_hot = tf.concat([input_dict_c["all_destinations"][:,:,0:2], new_probs], axis=-1)
            input_dict_c["all_destinations"] = dest_loc_one_hot

            predicted_rep_out = self.dl_trainer.predict_repetitively_dict(input_dict_c, scale_input_tensor=False,
            num_out_predictions=num_steps,variable_len_input=variable_len_input)

            # Add the result to the stack list
            stack_list.append(predicted_rep_out)
            # Add the destination to the destination list
            if self.dl_trainer.scaler is not None:
                locations_unsc = self.dl_trainer.scaler.scale_tensor(dest_loc_one_hot[:,:,0:2], "denormalize", "in")
                dest_loc_one_hot = tf.concat([locations_unsc, new_probs], axis=-1)
            destination_list.append(dest_loc_one_hot)

        assembled_output = tf.stack(stack_list, axis = 0) # axes ((batch,) destination, time, x/y)

        return assembled_output, destination_list, dest_prob_dict

    def remove_padding_vals(self, array, padding_val = 0.):
        ''' remove the padding zeros from an array '''
        array_c = np.array(array)

        # only works for 2d
        batch_dim_present = array_c.ndim == 3
        if batch_dim_present:
            array_c = np.squeeze(array_c, axis=0)
        
        i = 0
        for i in range(len(array_c) - 1, 0, -1):
            if not (array_c[i,0]==padding_val and array_c[i,1]==padding_val):
                break            

        array_c = array_c[:i+1]

        # put batch dimension back if it was removed
        if batch_dim_present:
            array_c = np.expand_dims(array_c, axis=0)

        return array_c

    @property
    def graph(self):
        return self._graph

    @property
    def dl_trainer(self):
        return self._dl_trainer
