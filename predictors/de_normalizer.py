''' 
Testing has shown that normalizing the data is an essential part to reach convergence.
This class calculates a data normalizer based on a certain set of training data.
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import expand_dims

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

    def scale_dict_f(self, data_dict, action):
        data_dict_c = dict(data_dict)
        xy_key = "in_xy"
        data_dict_c[xy_key] = self.scale_tensor(data_dict_c[xy_key], "denormalize", "in")
        if "all_points" in data_dict_c:
            locations = data_dict_c["all_points"][:, :, 0:2]
            locations = self.scale_tensor(locations, "denormalize", "in", dict(zip(["pos_x", "pos_y"],[0,1])))
            probs = tf.expand_dims(data_dict_c["all_points"][:,:,2],axis=-1)
            data_dict_c["all_points"] = tf.concat(
                [locations, probs], 
                axis=-1)
        if "all_destinations" in data_dict_c:
            locations = data_dict_c["all_destinations"][:, :, 0:2]
            locations = self.scale_tensor(locations, "denormalize", "in", dict(zip(["pos_x", "pos_y"],[0,1])))
            probs = tf.expand_dims(data_dict_c["all_destinations"][:,:,2],axis=-1)
            data_dict_c["all_destinations"] = tf.concat(
                [locations, probs], 
                axis=-1)

        return data_dict_c

    def scale_tensor(self, data, action, in_out, custom_scale_dict = None):
        ''' 
        (De)Normalize a matrix of TF data. Ordered col list is used to retrieve the 
        Custom scaling can be done by providing the custom scale dict.
        '''
        # Sanity check
        # if tf.math.greater(tf.size(tf.shape(data)), tf.constant(3)):
        #     raise ValueError("Data input in scale_tensor has %d dimensions, should have 3."%(tf.size(tf.shape(data)).numpy()))
        # Assert that action is either "normalize" or "denormalize"
        allowed_actions = ["normalize", "denormalize"]
        if action not in allowed_actions:
            raise ValueError("Invalid action. Expected one of: %s" % allowed_actions)

        # Assert that in_out is either "in" or "out"
        allowed_inout = ["in", "out"]
        if in_out not in allowed_inout:
            raise ValueError("Invalid in_out. Expected one of: %s" % allowed_inout)

        # Get dtype of input to correctly create the tensors later
        in_dtype = data.dtype

        # Get a list in correct order of mu's and std's to efficiently (de)normalize using TF
        my_inout_dict = None
        if custom_scale_dict is None:
            my_inout_dict = self.in_dict if in_out == "in" else self.out_dict
        else:
            my_inout_dict = custom_scale_dict

        mu_list = [None] * len(my_inout_dict)
        std_list = [None] * len(my_inout_dict)
        
        # for col_name in scale_col_list:
        for col_name in list(self.scale_dict.keys()):
            mu_list[my_inout_dict[col_name]] = self.scale_dict[col_name]["mu"]
            std_list[my_inout_dict[col_name]] = self.scale_dict[col_name]["std"]

            mu_list = [x if x is not None else 0. for x in mu_list]
            std_list = [x if x is not None else 1. for x in std_list]

        # Reshape the data to easily apply data across batch dimension
        if tf.math.equal(tf.size(tf.shape(data)), tf.constant(3)):
            o_shape = data.shape
            normed_data = tf.reshape(data, (o_shape[0]*o_shape[1], o_shape[2]))
        else:
            o_shape = data.shape # necessary for tf graph...
            normed_data = data
        # Create tf tensors from the premade lists
        mu_tensor = tf.constant(mu_list, dtype=in_dtype)
        std_tensor = tf.constant(std_list, dtype=in_dtype)
        # (de)normalize
        if action == "normalize":
            normed_data = tf.map_fn(lambda line: tf.math.subtract(line,mu_tensor), normed_data)
            normed_data = tf.map_fn(lambda line: tf.math.divide(line,std_tensor), normed_data)
        elif action == "denormalize":
            normed_data = tf.map_fn(lambda line: tf.math.multiply(line,std_tensor), normed_data)
            normed_data = tf.map_fn(lambda line: tf.math.add(line,mu_tensor), normed_data)
        # back to original shape with batch dimension
        if tf.math.equal(tf.size(tf.shape(data)), tf.constant(3)):
            normed_data = tf.reshape(normed_data, o_shape)

        return normed_data
    
    # def scale_dict(self, data, action, in_out, xy_tensor_key):
    #     normed_data = dict(data)
    #     normed_data[xy_tensor_key] = self.scale_tensor(data[xy_tensor_key], action, in_out)
        
    #     return normed_data

    def generate_noise_std_vect(self, std_orig_scale, noise_cols, ordered_col_list):
        '''
        We want to add some noise in the dataset pipeline, but need to take into account the used scale factors
        Therefore, make a vector of to be used std deviations based on standard deviations specified on 1m level.
        '''
        ordered_col_list_d = dict(zip(ordered_col_list, list(range(len(ordered_col_list)))))
        std_list = [None] * len(ordered_col_list)        
        
        # for col_name in scale_col_list:
        for col_name in noise_cols:
            try: 
                # if feature has been scaled
                std_list[ordered_col_list_d[col_name]] = std_orig_scale/self.scale_dict[col_name]["std"]
            except:
                # if featire has not been scaled
                std_list[ordered_col_list_d[col_name]] = std_orig_scale

        std_list = [x if x is not None else 0. for x in std_list]

        return std_list

    def quick_xy_normer(self, d):
        mu_tensor = tf.constant([self.scale_dict["pos_x"]["mu"], self.scale_dict["pos_y"]["mu"]], dtype=tf.float64)
        std_tensor = tf.constant([self.scale_dict["pos_x"]["std"], self.scale_dict["pos_y"]["std"]], dtype=tf.float64)
    
        normed_data = d["xy"]
        normed_data = tf.map_fn(lambda line: tf.math.subtract(line,mu_tensor), normed_data)
        normed_data = tf.map_fn(lambda line: tf.math.divide(line,std_tensor), normed_data)
        d["xy"] = normed_data
        return d

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
    # Create example data to test
    t00 = 20*tf.random.uniform((17,6,1))
    t01 = -2*tf.random.uniform((17,6,1))

    t1 = tf.random.uniform((17,6,2))
    t2 = tf.random.uniform((17,6,2))
    t3 = tf.random.uniform((17,6,2))

    in_dict = {"a": 0, "b": 1}

    data_dict = dict(zip(
    ["a", "b"], 
    [tf.reshape(t00,(-1)), tf.reshape(t01,(-1))]
    ))
    normer = DataDeNormalizer(data_dict, in_dict, in_dict)
    normed = normer.scale_tensor(t1, "normalize", "in")
    normed = normer.scale_tensor(normed, "denormalize", "in")

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

    # # Test the scaling of a TF batch
    # one_batch = np.transpose(np.vstack(list(data_dict.values())))
    # batch_list = [one_batch for i in range(bs)]
    # data_tensor = tf.constant(np.stack(batch_list))

    # normer = DataDeNormalizer(data_dict)
    # normed = normer.scale_tensor(data_tensor, list(data_dict.keys()), "normalize")
    # denormed = normer.scale_tensor(normed, list(data_dict.keys()), 'denormalize')

    # print("Tensor (de)normalization:")
    # print(normed)
    # print(denormed)

    # # Test the scaling of the dataframe
    # normer_df = DataDeNormalizer.from_dataframe(dataframe, ["a", "b", "c"])
    # normed_df = normer.scale_dataframe(dataframe, ["a", "b"], "normalize")
    # denormed_df = normer.scale_dataframe(normed_df, ["a", "b"], "denormalize")

    # print("Dataframe (de)normalization:")
    # print(dataframe)
    # print(normed_df)
    # print(denormed_df)

    # # Actual testing
    # assert np.all([denormed, data_tensor])
    # assert dataframe.equals(denormed_df)
    # #TODO: Write asserton for normed vector


    return None

if __name__ == '__main__':
    __test()