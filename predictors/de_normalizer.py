''' 
Testing has shown that normalizing the data is an essential part to reach convergence.
This class calculates a data normalizer based on a certain set of training data.
'''

import pandas as pd
import numpy as np
import tensorflow as tf

class DataDeNormalizer():
    def __init__(self, data_dict, in_dict, out_dict):
        scale_dict = dict()

        # extract the mean/std for each variable in order to create the scale_dict
        for data_name, data_array in zip(data_dict.keys(), data_dict.values()):
            mu = np.average(data_array)
            std = np.std(data_array)

            scale_dict[data_name] = dict(zip(['mu', 'std'], [mu, std]))

        self._scale_dict = scale_dict
        self._in_dict = in_dict
        self._out_dict = out_dict

    @classmethod
    def from_dataframe(cls, df, column_names, in_dict, out_dict):
        ''' 
        Initiate the class using a dataframe with data to train the (de)normalizer 
        '''        
        df_c = df.copy()
        data_dict = dict()
        for column_name in column_names:
            data_dict[column_name] = df_c[column_name].to_numpy()
        return cls(data_dict, in_dict, out_dict)      

    def scale_dataframe(self, dataframe, col_names, action):
        '''
        (De)Normalize the selected columns of a pandas dataframe (returns new dataframe).
        '''  
        # Assert that action is either "normalize" or "denormalize"
        allowed_actions = ["normalize", "denormalize"]
        if action not in allowed_actions:
            raise ValueError("Invalid action. Expected one of: %s" % allowed_actions)

        # Assert that the names columns are allowed
        for col_name in col_names:
            if not col_name in self.scaler_columns:
                raise ValueError("Invalid column name. Expected one of: %s" % self.scaler_columns)

        # Let's do some scaling now
        df_c = dataframe.copy()

        for col_name in col_names:
            scaler_vals = self.scale_dict[col_name]

            if action == "normalize":
                df_c[col_name] = (df_c[col_name] - scaler_vals["mu"]) / scaler_vals["std"]
            else:
                df_c[col_name] = (df_c[col_name] * scaler_vals["std"]) + scaler_vals["mu"]
        return df_c


    def scale_tensor(self, data, scale_col_list, action, in_out):
        ''' 
        (De)Normalize a matrix of TF data. Ordered col list is used to retrieve the 
        '''
        # Assert that action is either "normalize" or "denormalize"
        allowed_actions = ["normalize", "denormalize"]
        if action not in allowed_actions:
            raise ValueError("Invalid action. Expected one of: %s" % allowed_actions)

        # Assert that in_out is either "in" or "out"
        allowed_inout = ["in", "out"]
        if in_out not in allowed_inout:
            raise ValueError("Invalid in_out. Expected one of: %s" % allowed_inout)

        # Get a list in correct order of mu's and std's to efficiently (de)normalize using TF
        my_scale_dict = None
        my_scale_dict = self.in_dict if in_out == "in" else self.out_dict

        mu_list = [None] * len(my_scale_dict)
        std_list = [None] * len(my_scale_dict)
        
        for col_name in scale_col_list:

            mu_list[my_scale_dict[col_name]] = self.scale_dict[col_name]["mu"]
            std_list[my_scale_dict[col_name]] = self.scale_dict[col_name]["std"]

            mu_list = [x if x is not None else 0. for x in mu_list]
            std_list = [x if x is not None else 1. for x in std_list]

        # Reshape the data to easily apply data across batch dimension
        o_shape = data.shape
        normed_data = tf.reshape(data, (o_shape[0]*o_shape[1], o_shape[2]))
        # Create tf tensors from the premade lists
        mu_tensor = tf.constant(mu_list, dtype='double')
        std_tensor = tf.constant(std_list, dtype='double')
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

    @property
    def scaler_columns(self):
        return list(self.scale_dict.keys())

    @property
    def in_dict(self):
        return self._in_dict
    
    @property 
    def out_dict(self):
        return self._out_dict

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
    dataframe = pd.DataFrame({'a': data_a, 'b': data_b, 'c': data_c})

    # Test the scaling of a TF batch
    one_batch = np.transpose(np.vstack(list(data_dict.values())))
    batch_list = [one_batch for i in range(bs)]
    data_tensor = tf.constant(np.stack(batch_list))

    normer = DataDeNormalizer(data_dict)
    normed = normer.scale_tensor(data_tensor, list(data_dict.keys()), "normalize")
    denormed = normer.scale_tensor(normed, list(data_dict.keys()), 'denormalize')

    print("Tensor (de)normalization:")
    print(normed)
    print(denormed)

    # Test the scaling of the dataframe
    normer_df = DataDeNormalizer.from_dataframe(dataframe, ["a", "b", "c"])
    normed_df = normer.scale_dataframe(dataframe, ["a", "b"], "normalize")
    denormed_df = normer.scale_dataframe(normed_df, ["a", "b"], "denormalize")

    print("Dataframe (de)normalization:")
    print(dataframe)
    print(normed_df)
    print(denormed_df)

    # Actual testing
    assert np.all([denormed, data_tensor])
    assert dataframe.equals(denormed_df)
    #TODO: Write asserton for normed vector


    return None

if __name__ == '__main__':
    __test()