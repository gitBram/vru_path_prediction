import tensorflow as tf
import numpy as np
from .de_normalizer import DataDeNormalizer

''' Create train, test and verification TF data pipeline for training from a big datasource or multiple datasources'''

class TFDataSet():
    ''' Class serving as a structural element to save a dataset and its parameters '''
    def __init__(self, dataset_dict,
    bs, in_len, out_len,
    train_perc, test_perc, val_perc,
    in_dict, out_dict,
    x_col_name, y_col_name,
    shuffled, normer=None, scale_cols = None):
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
    
    @classmethod
    def init_as_fixed_length(cls, dataframe, do_shuffle = True, shuffle_buffer_size = 100,
                        batch_size = 5, normalize_data = True,
                        label_length = 1, # timesteps to be considered in in/out frame, stride with which data is windowed
                        train_perc = .7, test_perc = .2, val_perc = .1,
                        feature_list_in = ['pos_x', 'pos_y'], feature_list_out = ['pos_x', 'pos_y'], # all features that will be in in-/output
                        scale_list = ['pos_x', 'pos_y'],
                        id_col_name = "agent_id", x_col_name = 'pos_x', y_col_name = 'pos_y',
                        seq_in_length = None, seq_stride = None, noise_std = None, n_repeats = 1):
        '''
        Create fixed length dataset (seq_in_length and seq_stride not None) or simple zero padded dataset with full paths (seq_in_length and seq_stride None)
        Add noise to test dataset (noise_std is standard deviation in meter) if noise_std is not None
        '''
        # Get everything for getting correct in and output columns
        s_in, s_out = set(feature_list_in), set(feature_list_out)
        l_diff = list(s_out - s_in)
        l_comb = list(s_in) + l_diff

        l_comb_with_id = l_comb.copy()
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

        # print(l_comb_with_id)
        # print(in_dict)
        # print(out_dict)

        # Train Val Test
        n_rows = data.shape[0]
        n_train, n_val = int(train_perc*n_rows), int(val_perc*n_rows)
        # n_test = n_rows - n_train - n_val

        df_train = data.iloc[:n_train, :]
        df_val = data.iloc[n_train:(n_train+n_val), :]
        df_test = data.iloc[(n_train+n_val):, :]

        # Set everything up for data normalization
        normer = None
        if normalize_data:
            normer = DataDeNormalizer.from_dataframe(df_train, scale_list, in_dict, out_dict) # no cheating! Only use train for normalization

            df_train = normer.scale_dataframe(df_train, scale_list, "normalize")
            df_val = normer.scale_dataframe(df_val, scale_list, "normalize")
            df_test = normer.scale_dataframe(df_test, scale_list, "normalize")   
                       
        # Set everything up for adding noise
        noise_std_vector = None
        if normalize_data:
            noise_std_vector = normer.generate_noise_std_vect(noise_std, ['pos_x', 'pos_y'], headers_id_dropped)
        else:
            noise_std_vector = None
        
        ds_dict = dict()    

        ds_dict["train"] = cls.__df_to_ds(cls, df_train, id_col_name, label_length, in_col_nums, out_col_nums, do_shuffle, shuffle_buffer_size, batch_size, seq_in_length, seq_stride, noise_std_vector, n_repeats=n_repeats)
        ds_dict["test"] = cls.__df_to_ds(cls, df_test, id_col_name, label_length, in_col_nums, out_col_nums, do_shuffle, shuffle_buffer_size, batch_size, seq_in_length, seq_stride)
        ds_dict["val"] = cls.__df_to_ds(cls, df_val, id_col_name, label_length, in_col_nums, out_col_nums, do_shuffle, shuffle_buffer_size, batch_size, seq_in_length, seq_stride, noise_std_vector, n_repeats=n_repeats)
       
        return cls(ds_dict, batch_size, seq_in_length, label_length, train_perc, test_perc, val_perc,
        in_dict, out_dict, x_col_name, y_col_name, do_shuffle, normer=normer, scale_cols=scale_list)

    def __df_to_ds(self, df, id_col_name, lbl_length, in_col_nums, out_col_nums, shuffle, 
    shuffle_buffer_size, batch_size, seq_in_length, seq_stride, noise_std_vector=None, n_repeats=None):
        ''' 
        Go from a pd dataframe to a TF dataset 
        If seq_in_length is None or seq_stride is None => ds with fixed length created, otherwise zero padded
        If noise_std_vector is not None, noise is added
        '''
        ds = None
        df_c = df.copy()
        for i in df_c[id_col_name].unique():
            path_df = df_c.loc[df_c[id_col_name] == i]
            path_df = path_df.drop(columns = id_col_name)
            path_np = path_df.to_numpy()

            # Calculate destinations probabilities
            dest_prob = np.array([0., 0., .0, float(i)])

            if not (seq_in_length is None or seq_stride is None):
                def windower(a, window_size):
                    n = a.shape[0]
                    n_out_frames = (n-window_size+1)
                    if n >= window_size:
                        return np.stack(a[i:i+window_size] for i in range(0,n_out_frames))
                    else:
                        return np.array([])
                
                path_np = windower(path_np, seq_in_length + lbl_length)

                # if no windows were returned --> skip to next path
                if path_np.size == 0:
                    continue                
            ds_dict = dict()
            ds_dict["xy"]=path_np
            ds_dict["probs"]=np.tile(dest_prob, (len(path_np), 1))
            path_ds = tf.data.Dataset.from_tensors(ds_dict)
            
            # Add noise if wanted
            if noise_std_vector is not None:
                def add_noise(x):                
                    noise = tf.random.normal(shape=x["xy"].shape, mean=0., stddev=noise_std_vector, dtype=x["xy"].dtype)
                    x["xy"] = x["xy"] + noise
                    return x
                path_ds = path_ds.map(add_noise)
            
            try:
                # Add to dataset
                ds = ds.concatenate(path_ds)
            except:
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

        if not (seq_in_length is None or seq_stride is None):
            ds = ds.batch(batch_size, drop_remainder=True)
        else:
            ds = ds.padded_batch(batch_size, padding_values = None)#.prefetch(1)
            

        # this function does nothing for now
        '''
        def pad_or_trunc(t,k):
            return t,k

        ds = ds.map(pad_or_trunc)
        '''

        return ds.repeat(n_repeats)

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
            input_denormed = dict(input)
            input_denormed[xy_key] = self.normer.scale_tensor(input[xy_key], "denormalize", "in")
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