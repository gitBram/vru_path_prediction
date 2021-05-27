import tensorflow as tf
import numpy as np
from .de_normalizer import DataDeNormalizer

''' Create train, test and verification TF data pipeline for training from a big datasource or multiple datasources'''
# class TFDataSetCreator():
#     def __init__(self, df):
#         self._df = df.copy()

#     def create_datasets_fixed_len(self, train_perc, test_perc, val_perc, 
#     input_names, output_names,
#     label_len, output_len, stride, batch_size,
#     do_shuffle = True, shuffle_buffer_size = 10, normalize_data = True):


#         return None

#     def init_ds_full(self, do_shuffle = True, shuffle_buffer_size = 100,
#                         batch_size = 5, normalize_data = True,
#                         label_length = 1, # timesteps to be considered in in/out frame, stride with which data is windowed
#                         padding = True, padding_kind = 'front',
#                         cutting = False, cutting_kind = 'front',
#                         train_perc = .7, test_perc = .2, val_perc = .1,
#                         feature_list_in = ['pos_x', 'pos_y'], feature_list_out = ['pos_x', 'pos_y'], id_col = "agent_id"):  # the fixed length sequences in the dataset
#         ''' Function to initiate dataset with fixed lengths, based on splitting up dataframe per agent_id '''
        
#         # Get everything for getting correct in and output columns
#         s_in, s_out = set(feature_list_in), set(feature_list_out)
#         l_diff = list(s_out - s_in)
#         l_comb = list(s_in) + l_diff

#         l_comb_with_id = l_comb.copy()
#         l_comb_with_id.append(self.id_col_name)

#         data = self.df[l_comb_with_id]

#         headers_id_dropped = data.columns.to_list() # in order to keep column ids after id column drop
#         headers_id_dropped.remove(self.id_col_name)

#         # get in and out columns in correct order for correctly retrieving info from dataset
#         l_in = [x for x in headers_id_dropped if x in feature_list_in]
#         l_out = [x for x in headers_id_dropped if x in feature_list_out]

#         # get column numbers of output columns to later extract
#         out_col_nums = [headers_id_dropped.index(col) for col in l_out] 
#         in_col_nums = [headers_id_dropped.index(col) for col in l_in] 

#         # Dictionary for later use
#         in_dict = dict(zip(l_in, list(range(len(l_in))))) 
#         out_dict = dict(zip(l_out, list(range(len(l_out))))) 

#         # Train Val Test
#         n_rows = data.shape[0]
#         n_train, n_val = int(train_perc*n_rows), int(val_perc*n_rows)
#         # n_test = n_rows - n_train - n_val

#         df_train = data.iloc[:n_train, :]
#         df_val = data.iloc[n_train:(n_train+n_val), :]
#         df_test = data.iloc[(n_train+n_val):, :]
        
#         ds_dict = dict()    

#         ds_dict["train"] = self.__df_to_ds(df_train, id_col, label_length, in_col_nums, out_col_nums, do_shuffle, shuffle_buffer_size)
#         ds_dict["test"] = self.__df_to_ds(df_test, id_col, label_length, in_col_nums, out_col_nums, do_shuffle, shuffle_buffer_size)
#         ds_dict["val"] = self.__df_to_ds(df_val, id_col, label_length, in_col_nums, out_col_nums, do_shuffle, shuffle_buffer_size)
        
#         out_ds = TFDataSet(ds_dict, )

#         # tf_dataset = dict()
#         # tf_dataset["train"] = ds
#         # self.tf_dataset = tf_dataset
  
#     @property
#     def df(self):
#         return self._df



class TFDataSet():
    ''' Class serving as a structural element to save a dataset and its parameters '''
    def __init__(self, dataset_dict,
    bs, in_len, out_len,
    train_perc, test_perc, val_perc,
    in_dict, out_dict,
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
        self.bs = bs
        self.in_len = in_len
        self.out_len = out_len
        self.train_perc = train_perc
        self.test_perc = test_perc
        self.val_perc = val_perc
        self.in_dict = in_dict
        self.out_dict = out_dict
        self.shuffled = shuffled

        # Normer for conveniently denorming without effort
        self.normer = normer
        self.scale_cols = scale_cols

        # Saving the TF datasets
        self.tf_ds_dict = dataset_dict
    
    @classmethod
    def init_as_fixed_length(cls, dataframe, do_shuffle = True, shuffle_buffer_size = 100,
                        batch_size = 5, normalize_data = True,
                        input_length = 5, label_length = 1, # timesteps to be considered in in/out frame, stride with which data is windowed
                        train_perc = .7, test_perc = .2, val_perc = .1,
                        feature_list_in = ['pos_x', 'pos_y'], feature_list_out = ['pos_x', 'pos_y'], # all features that will be in in-/output
                        scale_list = ['pos_x', 'pos_y'],
                        id_col_name = "agent_id",
                        seq_in_length = None, seq_stride = None):

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
        
        ds_dict = dict()    

        ds_dict["train"] = cls.__df_to_ds(cls, df_train, id_col_name, label_length, in_col_nums, out_col_nums, do_shuffle, shuffle_buffer_size, batch_size, seq_in_length, seq_stride)
        ds_dict["test"] = cls.__df_to_ds(cls, df_test, id_col_name, label_length, in_col_nums, out_col_nums, do_shuffle, shuffle_buffer_size, batch_size, seq_in_length, seq_stride)
        ds_dict["val"] = cls.__df_to_ds(cls, df_val, id_col_name, label_length, in_col_nums, out_col_nums, do_shuffle, shuffle_buffer_size, batch_size, seq_in_length, seq_stride)
       
        return cls(ds_dict, batch_size, input_length, label_length, train_perc, test_perc, val_perc,
        in_dict, out_dict, do_shuffle, normer=normer, scale_cols=scale_list)

    def __df_to_ds(self, df, id_col_name, lbl_length, in_col_nums, out_col_nums, shuffle, 
    shuffle_buffer_size, batch_size, seq_in_length, seq_stride):
        ''' 
        Go from a pd dataframe to a TF dataset 
        If seq_in_length is None or seq_stride is None => ds with fixed length created, otherwise zero padded
        '''
        ds = None
        df_c = df.copy()
        for i in df_c[id_col_name].unique():
            path_df = df_c.loc[df_c[id_col_name] == i]
            path_df = path_df.drop(columns = id_col_name)
            path_np = path_df.to_numpy()

            if not (seq_in_length is None or seq_stride is None):
                def windower(a, window_size=3):
                    n = a.shape[0]
                    n_out_frames = (n-window_size+1)
                    if n >= window_size:
                        return np.stack(a[i:i+window_size] for i in range(0,n_out_frames))
                    else:
                        return np.array([])
                
                path_np = windower(path_np)

            path_ds = tf.data.Dataset.from_tensors(path_np)
            a = next(iter(path_ds)).numpy()

            try:
                # Add to dataset
                ds = ds.concatenate(path_ds)
            except:
                # Add first ds entry
                ds = path_ds

        # Let's remove the first dimension (*random*,timesteps, feature_num) to (timesteps, feature_num)
        ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        
        ds = ds.map(lambda window: (tf.transpose(tf.gather(tf.transpose(window[:-lbl_length]),in_col_nums)), tf.transpose(tf.gather(tf.transpose(window[-lbl_length:]),out_col_nums))))

        if shuffle == True: 
            ds = ds.shuffle(buffer_size=shuffle_buffer_size) 

        if not (seq_in_length is None or seq_stride is None):
            ds = ds.batch(batch_size)
        else:
            ds = ds.padded_batch(batch_size, padding_values = None)#.prefetch(1)
            

        # this function does nothing for now
        def pad_or_trunc(t,k):
            return t,k

        ds = ds.map(pad_or_trunc)

        return ds

    def example(self, ds_type, denorm):
        ''' Return an example batch from train/test/val dataset, denorm if wanted (and only if denorm available ofc) '''
        # Quick sanity check
        if not ds_type in list(self.tf_ds_dict.keys()):
            raise ValueError("Key %s not in dataset dict"%(ds_type))        
        if not type(denorm) is bool:
            raise ValueError("Expected bool, got %s"%(type(denorm)))
        
        input, output = next(iter(self.tf_ds_dict[ds_type]))
        if denorm and self.normer is not None:
            input_d = self.normer.scale_tensor(input, list(self.scale_cols), "denormalize", "in")
            output_d = self.normer.scale_tensor(output, list(self.scale_cols), "denormalize", "out")

            return input_d, output_d
        else: 
            return input, output

def __test():
    return None

if __name__ == '__main__':
    __test()