import tensorflow as tf
import numpy as np
from .de_normalizer import DataDeNormalizer
import pickle
import os
from tqdm import tqdm

''' Create train, test and verification TF data pipeline for training from a big datasource or multiple datasources'''

class TFDataSet():
    ''' Class serving as a structural element to save a dataset and its parameters '''
    def __init__(self, dataset_dict,
    bs, in_len, out_len,
    train_perc, test_perc, val_perc,
    in_dict, out_dict,
    x_col_name, y_col_name,
    shuffled, size_dict, normer=None, scale_cols = None):
        # quick sanity check
        allowed_keys = ["test", "train", "val"]
        
        if not len((dataset_dict.keys())) <= len(allowed_keys):
            raise ValueError("Too many keys in dataset dict.")
        if not len((dataset_dict.keys())) > 0:
            raise ValueError("Empty dataset dict.")
        for key in list(dataset_dict.keys()):
            if key not in allowed_keys:
                raise ValueError("Invalid key in dataset dict. Expected one of: %s" % allowed_keys)

        # Store these variables since it is convenient later on
        self._bs = bs
        self._in_len = in_len
        self._out_len = out_len
        self.train_perc = train_perc
        self.test_perc = test_perc
        self.val_perc = val_perc
        self._in_dict = in_dict
        self._out_dict = out_dict
        self.shuffled = shuffled

        # Normer for conveniently denorming without effort
        self.normer = normer
        self.scale_cols = scale_cols

        # Saving the TF datasets
        self._tf_ds_dict = dataset_dict

        # For creating generalised in and out dict
        self._x_col_name = x_col_name
        self._y_col_name = y_col_name

        # For later use in creation of model
        self.size_dict = size_dict

    @classmethod
    def init_as_fixed_length(cls, dataframe, graph, var_in_len, do_shuffle = True, shuffle_buffer_size = 100,
                        batch_size = 5, normalize_data = True,
                        label_length = 1, # timesteps to be considered in in/out frame
                        train_perc = .7, test_perc = .2, val_perc = .1,
                        feature_list_in = ['pos_x', 'pos_y'], feature_list_out = ['pos_x', 'pos_y'], # all features that will be in in-/output
                        scale_list = ['pos_x', 'pos_y'], 
                        id_col_name = "agent_id", x_col_name = 'pos_x', y_col_name = 'pos_y',
                        seq_in_length = None, length_stride = None, noise_std = None, n_repeats = 1,
                        extra_features_dict = None, 
                        ds_creation_dict = None, save_folder = None):
        '''
        Create fixed length dataset (seq_in_length not None) or variable length non-batched dataset (seq_in_length is None)
        Add noise to test dataset (noise_std is standard deviation in meter) if noise_std is not None
        Extra_features contains extra features to be added in dataset, graph is needed to get those extra features (destination probabilities etc)
        '''
        # Get everything for getting correct in and output columns
        s_in, s_out = set(feature_list_in), set(feature_list_out)
        l_diff = list(s_out - s_in)
        l_comb = list(s_in) + l_diff

        # Sort the features (otherwise not repeatable) and add the id for later filtering
        l_comb_with_id = l_comb.copy()
        l_comb_with_id = sorted(l_comb_with_id)
        l_comb_with_id.append(id_col_name)        

        data = dataframe.copy()[l_comb_with_id]        

        headers_id_dropped = data.columns.to_list() # in order to keep column ids after id column drop
        headers_id_dropped.remove(id_col_name)

        # Get in and out columns in correct order for correctly retrieving info from dataset
        l_in = [x for x in headers_id_dropped if x in feature_list_in]
        l_out = [x for x in headers_id_dropped if x in feature_list_out]

        # get column numbers of output columns to later extract
        out_col_nums = [headers_id_dropped.index(col) for col in l_out] 
        in_col_nums = [headers_id_dropped.index(col) for col in l_in] 

        # Dictionary for later use
        in_dict = dict(zip(l_in, list(range(len(l_in))))) 
        out_dict = dict(zip(l_out, list(range(len(l_out))))) 

        # Set everything up for data normalization
        normer = None
        if normalize_data:
            normer = DataDeNormalizer.from_dataframe(data, scale_list, in_dict, out_dict) # no cheating! Only use train for normalization
        
        # for norming
        # data = normer.scale_dataframe(data, scale_list, "normalize") # CANNOT NORM YET --> GRAPH ANALYSIS STILL TO HAPPEN
        noise_std_vect = normer.generate_noise_std_vect(0.3, ["pos_x","pos_y"], ["pos_x", "pos_y"])

        # Train Val Test
        n_rows = data.shape[0]
        n_train, n_val = int(train_perc*n_rows), int(val_perc*n_rows)
        # n_test = n_rows - n_train - n_val

        df_train = data.iloc[:n_train, :]
        df_val = data.iloc[n_train:(n_train+n_val), :]
        df_test = data.iloc[(n_train+n_val):, :]
        # if path analysis was not given, do it
        if ds_creation_dict is None:
            # Make sure the graph transition matrices are calculated
            graph.recalculate_trans_mat_dependencies()

            # Create the ds dicts
            ds_creation_dict = dict()
            ds_creation_dict["train"] = cls.__df_to_ds_dict(cls, df_train, id_col_name, label_length, seq_in_length, var_in_len, length_stride, extra_features_dict=extra_features_dict, graph=graph, normer=normer)
            ds_creation_dict["test"] = cls.__df_to_ds_dict(cls, df_test, id_col_name, label_length, seq_in_length, var_in_len, length_stride, extra_features_dict=extra_features_dict, graph=graph, normer=normer)
            ds_creation_dict["val"] = cls.__df_to_ds_dict(cls, df_val, id_col_name, label_length, seq_in_length, var_in_len, length_stride, extra_features_dict=extra_features_dict, graph=graph, normer=normer)
        # save the creation dicts for later use
            with open(save_folder, 'wb') as handle:
                pickle.dump(ds_creation_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # turn the path analysis into real ds dict
        ds_dict = dict()

        ds_dict["train"], size_dict = cls.ds_dict_to_ds(cls, ds_creation_dict["train"], normer, noise_std, 
        do_shuffle, shuffle_buffer_size, batch_size, n_repeats, label_length, in_col_nums, out_col_nums, noise_std_vect,
        var_in_len, length_stride)
        ds_dict["test"], _ = cls.ds_dict_to_ds(cls, ds_creation_dict["test"], normer, noise_std, 
        do_shuffle, shuffle_buffer_size, batch_size, n_repeats, label_length, in_col_nums, out_col_nums, noise_std_vect,
        var_in_len, length_stride)
        ds_dict["val"], _ = cls.ds_dict_to_ds(cls, ds_creation_dict["val"], normer, noise_std, 
        do_shuffle, shuffle_buffer_size, batch_size, n_repeats, label_length, in_col_nums, out_col_nums, noise_std_vect,
        var_in_len, length_stride)

        return cls(ds_dict, batch_size, seq_in_length, label_length, train_perc, test_perc, val_perc,
        in_dict, out_dict, x_col_name, y_col_name, do_shuffle, size_dict, normer=normer, scale_cols=scale_list)

    def __df_to_ds_dict(self, df, id_col_name, lbl_length, seq_in_length, var_in_len, length_stride,
    extra_features_dict=None, graph=None, normer=None):
        ''' 
        Go from a pd dataframe to a TF dataset 
        If seq_in_length is None or seq_stride is None => ds with fixed length created, otherwise zero padded
        If noise_std_vector is not None, noise is added
        '''
        # inits
        ds = None
        df_c = df.copy()
        max_label_len = df_c.groupby(id_col_name).count().max()[0]

        # Sanity check
        if extra_features_dict is not None:
            extra_f_allowed = ["all_points", "all_destinations", "n_connected_points", "n_destinations", "n_points"]
            extra_f_allowed2 = [f_allowed + "_after" for f_allowed in extra_f_allowed]
            extra_f_allowed = extra_f_allowed + extra_f_allowed2
            for ef in list(extra_features_dict.keys()):
                if not ef in extra_f_allowed:
                    raise ValueError("Extra features should be one of following:%s."%(extra_f_allowed))

        # Extracting paths from dataframe
        ds_pickle_list = []

        for i in tqdm(df_c[id_col_name].unique()):
            # create a dictionary for this path, will eventually be used for TF DS
            ds_dict = dict()            
            # extract a dataframe of one trajectory
            path_df = df_c.loc[df_c[id_col_name] == i]
            path_df = path_df.drop(columns = id_col_name)
            path_np = path_df.to_numpy()

            # for getting variable output length
            #   get number of data points in this path
            num_points = len(path_df)

            if num_points-lbl_length+1 < seq_in_length:
                print("Num points %d and lbl length %d and seq in length %d" % (len(path_np), lbl_length, seq_in_length))
                continue
            
            if not var_in_len:
                seq_in_length_l = [seq_in_length]   
            else:
                seq_in_length_l = range(seq_in_length, num_points-lbl_length+1, length_stride)   

            for in_length in seq_in_length_l:
                # Windowing if needed

                def windower(a, window_size):
                    n = a.shape[0]
                    n_out_frames = max((n-window_size+1),0)
                    if n >= window_size:
                        return np.stack(a[i:i+window_size] for i in range(0,n_out_frames)), n_out_frames
                    else:
                        return np.array([]), n_out_frames


                # windowing with fixed in put length                  
                path_np_w, n_frames = windower(path_np, in_length + lbl_length)

                # if no windows were returned --> skip to next path
                if path_np_w.size == 0:
                    print("continue2")
                    continue    

                ds_dict["xy"]=path_np_w

                # include the full input and output path for later analysis
                #   Create the lists
                label_list = []
                for i in range(n_frames):
                    label_list.append(path_np[in_length+i:,:])
                full_input_list = []
                for i in range(n_frames):
                    full_input_list.append(path_np[:in_length+i,:])

                #   Add to dict
                ds_dict["labels"]=tf.ragged.constant(label_list).to_tensor()
                lbl_len = ds_dict["labels"].shape[0]
                the_type_out = ds_dict["labels"].dtype

                ds_dict["input_labels"]=tf.ragged.constant(full_input_list).to_tensor()
                lbl_len = ds_dict["input_labels"].shape[0]
                the_type_in = ds_dict["input_labels"].dtype

                # if error is raised here, give it a bigger max label length
                ds_dict["labels"] = tf.concat([ds_dict["labels"], tf.zeros([lbl_len, max_label_len-lbl_len, 2], dtype=the_type_out)], axis=-2)
                ds_dict["input_labels"] = tf.concat([ds_dict["input_labels"], tf.zeros([lbl_len, max_label_len-lbl_len, 2], dtype=the_type_in)], axis=-2)


                # include the wanted extra features
                extra_f_sizes = None
                if not extra_features_dict is None:
                    before_or_after_split = "before"
                    extra_f_sizes = dict()
                    # ["all_points", "all_destinations", "n_connected_points", "n_points", "n_destinations"]
                    extra_f_keys = list(extra_features_dict.keys())

                    if "all_points" in extra_f_keys:
                        extra_f = "all_points" 
                        ds_dict[extra_f] = self.__get_point_prob_tensor(self=self, path=path_np, normer=normer, graph=graph, points_or_dests="points")
                        # Multiply the extra feature for each window
                        ds_dict[extra_f] = tf.tile(tf.expand_dims(ds_dict[extra_f],0), [n_frames, 1, 1])
                    if "all_destinations" in extra_f_keys:
                        extra_f = "all_destinations"
                        ds_dict[extra_f] = self.__get_point_prob_tensor(self=self, path=path_np, normer=normer, graph=graph, points_or_dests="destinations")
                        # Multiply the extra feature for each window
                        ds_dict[extra_f] = tf.tile(tf.expand_dims(ds_dict[extra_f],0), [n_frames, 1, 1])
                    if "n_connected_points" in extra_f_keys:
                        extra_f = "n_connected_points"
                    if "n_destinations" in extra_f_keys:
                        extra_f = "n_destinations"
                        ds_dict[extra_f] = self.__get_n_point_prob_tensor(self, path_np, normer, graph, 
                        extra_features_dict[extra_f], "destinations")
                        # Multiply the extra feature for each window
                        ds_dict[extra_f] = tf.tile(tf.expand_dims(ds_dict[extra_f],0), [n_frames, 1, 1])
                    if "n_points" in extra_f_keys:
                        extra_f = "n_points"
                        ds_dict[extra_f] = self.__get_n_point_prob_tensor(self, path_np, normer, graph, 
                        extra_features_dict[extra_f], "points")
                        # Multiply the extra feature for each window
                        ds_dict[extra_f] = tf.tile(tf.expand_dims(ds_dict[extra_f],0), [n_frames, 1, 1])

                    for path in path_np_w:
                        # no cheating: only get the points that will be used as input
                        path = path[:-lbl_length, :]
                        if "all_points_after" in extra_f_keys:
                            extra_f = "all_points_after" 
                            # Get the feature output
                            f_result = tf.expand_dims(
                                self.__get_point_prob_tensor(self=self, path=path, normer=normer, graph=graph, points_or_dests="points"),
                                axis=0)
                            # Add the feature output to the dict
                            if extra_f in ds_dict:
                                ds_dict[extra_f] = tf.concat([ds_dict[extra_f],f_result], axis=0)
                            else:
                                ds_dict[extra_f] = f_result

                        if "all_destinations_after" in extra_f_keys:
                            extra_f = "all_destinations_after"
                            f_result = tf.expand_dims(
                                self.__get_point_prob_tensor(self=self, path=path, normer=normer, graph=graph, points_or_dests="destinations"),
                                axis=0)
                            # Add the feature output to the dict
                            if extra_f in ds_dict:
                                ds_dict[extra_f] = tf.concat([ds_dict[extra_f],f_result], axis=0)
                            else:
                                ds_dict[extra_f] = f_result
                            # Save the shape of the feature for setting up the neural network
                            extra_f_sizes[extra_f] = ds_dict[extra_f].shape
                        if "n_connected_points_after" in extra_f_keys:
                            extra_f = "n_connected_points_after"
                        if "n_destinations_after" in extra_f_keys:
                            extra_f = "n_destinations_after"
                            f_result = tf.expand_dims(
                                self.__get_n_point_prob_tensor(self, path, normer, graph, extra_features_dict[extra_f], "destinations"),
                                axis=0)
                            # Add the feature output to the dict
                            if extra_f in ds_dict:
                                ds_dict[extra_f] = tf.concat([ds_dict[extra_f],f_result], axis=0)
                            else:
                                ds_dict[extra_f] = f_result
                            # Save the shape of the feature for setting up the neural network
                            extra_f_sizes[extra_f] = ds_dict[extra_f].shape
                        if "n_points_after" in extra_f_keys:
                            extra_f = "n_points_after"
                            f_result = tf.expand_dims(
                                self.__get_n_point_prob_tensor(self, path, normer, graph, extra_features_dict[extra_f], "points"),
                                axis=0)
                            # Add the feature output to the dict
                            if extra_f in ds_dict:
                                ds_dict[extra_f] = tf.concat([ds_dict[extra_f],f_result], axis=0)
                            else:
                                ds_dict[extra_f] = f_result
                            # Save the shape of the feature for setting up the neural network
                            extra_f_sizes[extra_f] = ds_dict[extra_f].shape

                if not normer is None:
                    ds_dict["xy"] = normer.scale_tensor(ds_dict["xy"], "normalize", "in")
                    
                ds_pickle_list.append(ds_dict)

        return ds_pickle_list

    def ds_dict_to_ds(self, ds_dicts_list, normer, noise_std, 
    shuffle, shuffle_buffer_size, batch_size, n_repeats,
    lbl_length, in_col_nums, out_col_nums, noise_std_vector, var_in_len, length_stride):
        size_dict = dict()
        ex_ds_dict = ds_dicts_list[0]
        for key in list(ex_ds_dict.keys()):
            size_dict[key] = ex_ds_dict[key].shape
        
        # function for adding noise
        @tf.function
        def add_noise(x):  
          x_c = dict(x)
          if noise_std is not None:              
              noise = tf.random.normal(shape=x_c["xy"].shape, mean=0., stddev=noise_std, dtype=tf.float32)
              x_c["xy"] = x_c["xy"] + noise
              return x

        # function for norming data
        mu_tensor = tf.constant([normer.scale_dict["pos_x"]["mu"], normer.scale_dict["pos_y"]["mu"]], dtype=tf.float32)
        std_tensor = tf.constant([normer.scale_dict["pos_x"]["std"], normer.scale_dict["pos_y"]["std"]], dtype=tf.float32)       
        @tf.function
        def my_quick_xy_normer(d):
            d_c = dict(d)        
            normed_data = d_c["xy"]
            normed_data = tf.map_fn(lambda line: tf.math.subtract(line,mu_tensor), normed_data)
            normed_data = tf.map_fn(lambda line: tf.math.divide(line,std_tensor), normed_data)
            d_c["xy"] = normed_data
            return d_c

        # second function for norming data
            
        def add_noise2(d):     
          if noise_std_vector is not None:  
            d_c=dict(d)         
            noise = tf.random.normal(shape=d_c["xy"].shape, mean=0., stddev=noise_std_vector, dtype=d_c["xy"].dtype)
            d_c["xy"] = d_c["xy"] + noise
            return d_c          

        ds = None
        for ds_dict in ds_dicts_list:
            # Scaling if needed
            for key in ds_dict.keys():
                ds_dict[key]=tf.cast(ds_dict[key], dtype=tf.float32)

            path_ds = tf.data.Dataset.from_tensors(ds_dict)  
            
            # Add noise if wanted
            '''
            path_ds = path_ds.map(add_noise)
            '''
            path_ds = path_ds.map(add_noise2)
            
            # Norm if needed
            '''
            if normer is not None:
                path_ds = path_ds.map(lambda x: my_quick_xy_normer(x))  
            '''

            if ds is not None:
                # Add to dataset
                ds = ds.concatenate(path_ds)
            else:
                # Add first ds entry
                ds = path_ds

        # Let's remove the first dimension (*random*,timesteps, feature_num) to (timesteps, feature_num)

        ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))                
        
        def extract_columns(d):           
            d["in_xy"]= tf.transpose(tf.gather(tf.transpose(d["xy"][:-lbl_length]),in_col_nums))
            out = tf.transpose(tf.gather(tf.transpose(d["xy"][-lbl_length:]),out_col_nums))
            del d["xy"]
            return d, out
        
        ds = ds.map(lambda d: extract_columns(d))
        
        if shuffle == True: 
            ds = ds.shuffle(buffer_size=shuffle_buffer_size) 

        # if not (type(seq_in_length)==list or seq_stride is None):
        #     
        ds = ds.batch(batch_size)
        # else:

        # ds = ds.padded_batch(batch_size, padding_values = tf.constant(-100.,dtype=tf.float64))

        return ds.repeat(n_repeats).prefetch(10), size_dict

    def __get_n_point_prob_tensor(self, path, normer, graph, n, points_or_dests = "points"):
        # sanity check
        allowed_pos = ["points", "destinations"]
        if not points_or_dests in allowed_pos:
            raise ValueError("Input %s is not allowed. Only values %s are allowed."%(points_or_dests, allowed_pos))

        nodes, _ = graph.analyse_full_signal(path, False)
        if len(nodes) > 0:
            location_list, prob_list = graph.return_n_most_likely_points(n, nodes, points_or_dests=points_or_dests)
  
            loc_tensor = tf.constant(location_list, dtype=tf.float32)
            if normer is not None:
                loc_tensor = normer.scale_tensor(loc_tensor, "normalize", "in", dict(zip(["pos_x", "pos_y"],[0,1])))
            prob_tensor = tf.reshape(tf.constant(prob_list, dtype=tf.float32),[-1, 1])
            return tf.concat([loc_tensor, prob_tensor], axis=1)
        else:
            node,_ = graph.find_closest_point(path[-1])
            location_list, prob_list = graph.return_n_most_likely_points(n, node, points_or_dests=points_or_dests)
            loc_tensor = tf.constant(location_list, dtype=tf.float32)
            prob_tensor = tf.constant(prob_list, dtype=tf.float32)
            if normer is not None:
                loc_tensor = normer.scale_tensor(loc_tensor, "normalize", "in", dict(zip(["pos_x", "pos_y"],[0,1])))
            prob_tensor = tf.reshape(tf.constant(prob_list, dtype=tf.float32),[-1, 1])
            return tf.concat([loc_tensor,prob_tensor], axis=1)

    def __get_point_prob_tensor(self, path, normer, graph, points_or_dests = "points"):
        # sanity check
        allowed_pos = ["points", "destinations"]
        if not points_or_dests in allowed_pos:
            raise ValueError("Input %s is not allowed. Only values %s are allowed."%(points_or_dests, allowed_pos))

        nodes, _ = graph.analyse_full_signal(path, False)
        if len(nodes) > 0:
            out = graph.calculate_destination_probs(nodes, dests_or_points=points_or_dests)
            
            location_list = []
            prob_list = []
            for key, prob in zip(out.keys(), out.values()):
                location_list.append(graph.points_dict[key])
                prob_list.append(prob)
            loc_tensor = tf.constant(location_list, dtype=tf.float32)
            prob_tensor = tf.constant(prob_list, dtype=tf.float32)
            if normer is not None:
                loc_tensor = normer.scale_tensor(loc_tensor, "normalize", "in", dict(zip(["pos_x", "pos_y"],[0,1])))
            prob_tensor = tf.reshape(tf.constant(prob_list, dtype=tf.float32),[-1, 1])
            return tf.concat([loc_tensor, prob_tensor], axis=1)
        else:
            # Option 1: Just find probs from closest point (even though not in predefined range)
            node,_ = graph.find_closest_point(path[-1])
            out = graph.calculate_destination_probs(node, dests_or_points=points_or_dests)
            
            location_list = []
            prob_list = []
            for key, prob in zip(out.keys(), out.values()):
                location_list.append(graph.points_dict[key])
                prob_list.append(prob)
            loc_tensor = tf.constant(location_list, dtype=tf.float32)
            prob_tensor = tf.constant(prob_list, dtype=tf.float32)
            if normer is not None:
                loc_tensor = normer.scale_tensor(loc_tensor, "normalize", "in", dict(zip(["pos_x", "pos_y"],[0,1])))
            prob_tensor = tf.reshape(tf.constant(prob_list, dtype=tf.float32),[-1, 1])
            return tf.concat([loc_tensor,prob_tensor], axis=1)
            # Option 2: Returning zeros
            '''
            if points_or_dests == "points":
                location_list = graph.points_locations
            elif points_or_dests == "destinations":
                location_list = graph.destination_locations

            loc_tensor = tf.constant(location_list, dtype=tf.float32)
            if normer is not None:
                loc_tensor = normer.scale_tensor(loc_tensor, "normalize", "in", dict(zip(["pos_x", "pos_y"],[0,1])))
            return tf.concat([loc_tensor,
            tf.zeros((len(location_list), 1), dtype=tf.float32)], axis=1)
            '''


    def example(self, ds_type):
        ''' 
        Return an example batch from train/test/val dataset, denorm if wanted (and only if denorm available ofc) 
        Only if in and output are both xy tensors, no extra information may be included
        '''
        # Quick sanity check
        if not ds_type in list(self.tf_ds_dict.keys()):
            raise ValueError("Key %s not in dataset dict"%(ds_type))        
        
        input, output = next(iter(self.tf_ds_dict[ds_type]))
        if self.normer is not None:
            # If there has been scaling, return both unscaled and scaled version
            input_denormed = self.normer.scale_tensor(input, "denormalize", "in")
            output_denormed = self.normer.scale_tensor(output, "denormalize", "out")

            return (input, output), (input_denormed, output_denormed)
        else: 
            # If there has not been any scaling, 
            return (input, output), (input, output)

    def remove_padding_vals(array, padding_val = 0.):
        ''' remove the padding zeros from an array '''
        array_c = np.array(array)
        i = 0
        for i in range(len(array_c) - 1, 0, -1):
            if not (array_c[i,0]==padding_val and array_c[i,1]==padding_val):
                break    

        return array_c[:i+1]

    def example_dict(self, ds_type, xy_key):
        '''
        Like example function, but if input is specified as a dict
        '''
        # Quick sanity check
        if not ds_type in list(self.tf_ds_dict.keys()):
            raise ValueError("Key %s not in dataset dict"%(ds_type))        
        input, output = next(iter(self.tf_ds_dict[ds_type]))
        if self.normer is not None:
            # If there has been scaling, return both unscaled and scaled version
            input_denormed = self.normer.scale_dict_f(input, "denormalize")
            output_denormed = self.normer.scale_tensor(output, "denormalize", "out")

            return (input, output), (input_denormed, output_denormed)
        else: 
            # If there has not been any scaling, 
            return (input, output), (input, output)

    @property
    def tf_ds_dict(self):
        return self._tf_ds_dict
    @property
    def batch_size(self):
        return self._bs
    @property
    def num_in_steps(self):
        return self._in_len
    @property
    def num_out_steps(self):
        return self._out_len
    @property
    def in_dict(self):
        return self._in_dict
    @property
    def out_dict(self):
        return self._out_dict
    @property
    def num_in_features(self):
        return len(self.in_dict)
    @property
    def num_out_features(self):
        return len(self.out_dict)
    @property
    def x_col_name(self):
        return self._x_col_name
    @property
    def y_col_name(self):
        return self._y_col_name
    @property
    def generalised_in_dict(self):
        d = dict()
        d['x']=self.in_dict[self.x_col_name]
        d['y']=self.in_dict[self.y_col_name]
        return d
    @property
    def generalised_out_dict(self):
        d = dict()
        d['x']=self.out_dict[self.x_col_name]
        d['y']=self.out_dict[self.y_col_name]
        return d
        

    

def __test():
    return None

if __name__ == '__main__':
    __test()