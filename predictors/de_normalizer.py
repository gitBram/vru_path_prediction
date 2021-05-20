''' 
Testing has shown that normalizing the data is an essential part to reach convergence.
This class calculates a data normalizer based on a certain set of training data.
'''

import pandas as pd
import numpy as np
import tensorflow as tf

class DataDeNormalizer():
    def __init__(self, data_dict):
        scale_dict = dict()

        # extract the mean/std for each variable in order to create the scale_dict
        for data_name, data_array in zip(data_dict.keys(), data_dict.values()):
            mu = np.average(data_array)
            std = np.std(data_array)

            scale_dict[data_name] = dict(zip(['mu', 'std'], [mu, std]))

        self._scale_dict = scale_dict

    @classmethod
    def from_dataframe(cls, df, column_names):
        ''' initiate the class using a dataframe with data to train the (de)normalizer '''
        data_dict = dict()
        for column_name in column_names:
            data_dict[column_name] = df[column_name].to_numpy()
        return cls(data_dict)        

    def scale(self, data, ordered_col_list, action):
        ''' 
        (De)Normalize a matrix of TF data. Ordered col list is used to retrieve the 
        '''
        # Assert that action is either "normalize" or "denormalize"
        allowed_actions = ["normalize", "denormalize"]
        if action not in allowed_actions:
            raise ValueError("Invalid action. Expected one of: %s" % allowed_actions)

        # Get a list in correct order of mu's and std's to efficiently (de)normalize using TF
        mu_list = []
        std_list = []
        
        for col_name in ordered_col_list:
            mu_list.append(self.scale_dict[col_name]["mu"])
            std_list.append(self.scale_dict[col_name]["std"])


        # Reshape the data to easily apply data across batch dimension
        o_shape = data.shape
        normed_data = tf.reshape(data, (o_shape[0]*o_shape[1], o_shape[2]))
        # Create tf tensors from the premade lists
        mu_tensor = tf.constant(mu_list)
        std_tensor = tf.constant(std_list)
        # (de)normalize
        if action == "normalize":
            normed_data = tf.map_fn(lambda line: tf.math.subtract(line,mu_tensor), normed_data)
            normed_data = tf.map_fn(lambda line: tf.math.divide(line,std_tensor), normed_data)
        elif action == "denormalize":
            normed_data = tf.map_fn(lambda line: tf.math.multiply(line,std_tensor), normed_data)
            normed_data = tf.map_fn(lambda line: tf.math.add(line,mu_tensor), normed_data)
        # back to original shape with batch dimension
        normed_data = tf.reshape(normed_data, o_shape)

        return normed_data



    # Getters and Setters
    @property
    def scale_dict(self):
        return self._scale_dict

def __test():
    # create example data to test
    n_rep = 3
    bs = 2
    data_a = [1., 2., 3.]
    data_b = [4., 5., 6.]
    data_c = [7., 8., 9.]

    data_dict = dict(zip(
        ["a", "b", "c"], 
        [np.repeat(data_a, n_rep), np.repeat(data_b, n_rep), np.repeat(data_c, n_rep)]
        ))

    one_batch = np.transpose(np.vstack(list(data_dict.values())))
    batch_list = [one_batch for i in range(bs)]
    data_tensor = tf.constant(np.stack(batch_list))

    normer = DataDeNormalizer(data_dict)
    normed = normer.scale(data_tensor, list(data_dict.keys()), "normalize")
    denormed = normer.scale(normed, list(data_dict.keys()), 'denormalize')

    print(normed)
    print(denormed)


    # Actual testing
    assert np.all([denormed, data_tensor])

    #TODO: Write asserton for normed vector


    return None

if __name__ == '__main__':
    __test()