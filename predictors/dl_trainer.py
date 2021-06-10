''' class for training deep learning models and setting parameters for the training phase'''

import tensorflow as tf
from tensorflow.python import keras
from keras.layers.core import Lambda
from keras import backend as K
from keras import Model
from keras.layers import Dense, LSTM, Reshape, InputLayer, Flatten, Concatenate, Dropout, Concatenate, Input

class DLTrainer:
    def __init__(self, max_epochs, patience, loss_function = tf.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(), metric = tf.metrics.MeanAbsoluteError()):
        self._max_epochs = max_epochs
        self._patience = patience
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._metric = metric

        self._model = None
        self._scaler = None
        self._num_in_features = None
        self._num_out_features = None
        self._num_in_steps = None
        self._num_out_steps = None


    def compile_and_fit(self, ds_creator_inst, save_path = None):
        dataset_dict = ds_creator_inst.tf_ds_dict
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=self._patience,
                                                        mode='min',
                                                        restore_best_weights=True)

        self.model.compile(loss=self._loss_function,
                    optimizer=self._optimizer,
                    metrics=[self._metric])
        cb = [early_stopping]

        history = self.model.fit(dataset_dict['train'], epochs=self.max_epochs,
                            validation_data=dataset_dict['val'],
                            callbacks=cb)
        if save_path is not None:
            self.model.save_weights(save_path)        

        return history

    def load_weights(self, save_path):
        # do a prediction in order to initialize the model...
        # self.model(tf.zeros((1, self.num_in_steps, self.num_in_features)))
        # now really load the weights
        self.model.load_weights(save_path)

    def predict(self, input_tensor, scale_input_tensor, n_evaluations = 1):
        ''' 
        Use the model to do a prediction
        '''
        # check that input is 3D (batch - time - feature)
        if len(input_tensor.shape) <= 2:
            input_tensor = tf.reshape(input_tensor, [1, -1, self.num_in_features])
        
        # sanity check
        if n_evaluations < 1:
            raise ValueError("Number of evaluations should be 1 or larger. %d was received." % (n_evaluations))

        if not (type(n_evaluations)==int):
            raise ValueError("Number of evaluations should be of type int. %s type was received." % (type(n_evaluations)))
                
        composed_output = composed_output_s = None

        for i in range(n_evaluations):
            # scale input tensor if requested, get prediction
            if scale_input_tensor:
                output = self.model(self.scaler.scale_tensor(input_tensor, "normalize", "in"))
            else:
                output = self.model(input_tensor)

            # scale output tensor if needed
            if self.scaler is not None:
                output_s = self.scaler.scale_tensor(output, "denormalize", "out")
            else:
                output_s = output
            
            # aggregate the outputs, only useful for epistemic 
            try:
                composed_output_s = tf.concat([composed_output_s,output_s], axis=1)
                composed_output = tf.concat([composed_output,output], axis=1)
            except:
                composed_output_s = output_s
                composed_output = output

        return composed_output, composed_output_s     

    def predict_dict(self, input_dict, scale_input_tensor, n_evaluations = 1):
        ''' 
        Use the model to do a prediction based on a dictionary model input
        '''
        input_dict_c = dict(input_dict)
        # check that input is 3D (batch - time - feature)
        if len(input_dict_c["in_xy"].shape) <= 2:
            input_dict_c["in_xy"] = tf.reshape(input_dict_c["in_xy"], [1, -1, self.num_in_features])
        
        # sanity check
        if n_evaluations < 1:
            raise ValueError("Number of evaluations should be 1 or larger. %d was received." % (n_evaluations))

        if not (type(n_evaluations)==int):
            raise ValueError("Number of evaluations should be of type int. %s type was received." % (type(n_evaluations)))
                
        composed_output = composed_output_s = None

        for i in range(n_evaluations):
            # scale input tensor if requested, get prediction
            if scale_input_tensor:
                output = self.model(self.scaler.scale_dict_f(input_dict_c, "normalize"))
            else:
                output = self.model(input_dict_c)

            # scale output tensor if needed
            if self.scaler is not None:
                output_s = self.scaler.scale_tensor(output, "denormalize", "out")
            else:
                output_s = output
            
            # aggregate the outputs, only useful for epistemic 
            try:
                composed_output_s = tf.concat([composed_output_s,output_s], axis=1)
                composed_output = tf.concat([composed_output,output], axis=1)
            except:
                composed_output_s = output_s
                composed_output = output

        return composed_output, composed_output_s  

    def predict_repetitively_dict(self, input_dict, scale_input_tensor, num_repetitions, fixed_len_input):
        input_dict_c = dict(input_dict)
        # Make sure input tensor is 3d 
        if len(input_dict_c["in_xy"].shape) <= 2:
            input_dict_c["in_xy"] = tf.expand_dims(input_dict_c["in_xy"], axis=0)

        # Get type same with output of model (float instead of double) to be able to concat
        input_dict_c["in_xy"] = tf.cast(input_dict_c["in_xy"], dtype=tf.float32)
        assembled_output = None
        for i in range(num_repetitions):
            # get one prediction
            new_output, new_output_s = self.predict_dict(input_dict_c, scale_input_tensor)

            try:
                assembled_output = tf.concat([assembled_output, new_output_s], axis=1)
            except:
                assembled_output = new_output_s

            # add the output to the new input, drop first of input in case it input should be kept constant length
            input_dict_c["in_xy"] = tf.concat([input_dict_c["in_xy"], new_output], axis=1)

            # drop the n first rows if input has to stay same length
            if fixed_len_input:
                input_dict_c["in_xy"] = input_dict_c["in_xy"][:, -self.num_in_steps:, :]

        return assembled_output

    def predict_repetitively(self, input_tensor, scale_input_tensor, num_repetitions, fixed_len_input):
        # Make sure input tensor is 3d 
        if len(input_tensor.shape) <= 2:
            input_tensor = tf.expand_dims(input_tensor, axis=0)

        # Get type same with output of model (float instead of double) to be able to concat
        assembled_input = tf.cast(input_tensor, dtype=tf.float32)
        assembled_output = None
        for i in range(num_repetitions):
            # get one prediction
            new_output, new_output_s = self.predict(assembled_input, scale_input_tensor)

            try:
                assembled_output = tf.concat([assembled_output, new_output_s], axis=1)
            except:
                assembled_output = new_output_s

            # add the output to the new input, drop first of input in case it input should be kept constant length
            assembled_input = tf.concat([assembled_input, new_output], axis=1)

            # drop the n first rows if input has to stay same length
            if fixed_len_input:
                assembled_input = assembled_input[:, -self.num_in_steps:, :]

        return assembled_output

    def predict_repetitively_epi(self, input_tensor, scale_input_tensor, num_repetitions, fixed_len_input, max_subsample_points):
        '''
        Predict one step at a time, epistemic uncertainty
        '''
        # predict 10
        

        return None

    def LSTM_one_shot_predictor(self, ds_creator_inst, lstm_layer_size, dense_layer_size, n_LSTM_layers, n_dense_layers):
        in_xy = Input(shape=(ds_creator_inst.num_in_steps, ds_creator_inst.num_in_features), name = 'in_xy')
        m = LSTM(lstm_layer_size, return_sequences=True)(in_xy)
        
        for i in range(n_LSTM_layers - 1):
            m = LSTM(lstm_layer_size, return_sequences=False)(m)

        for i in range(n_dense_layers):
            m = Dense(dense_layer_size)(m)
        
        m = Dense(ds_creator_inst.num_out_steps*ds_creator_inst.num_out_features,
                                    kernel_initializer=tf.initializers.random_uniform)(m)
        m = Reshape([ds_creator_inst.num_out_steps, ds_creator_inst.num_out_features])(m)

        model = tf.keras.Model(
            [in_xy],
            [m]
        )

        self.model = model

        # copy the scaler
        self.scaler = ds_creator_inst.normer

        # set some helper vars
        self.num_in_features = ds_creator_inst.num_in_features
        self.num_out_features = ds_creator_inst.num_out_features

        self.num_in_steps = ds_creator_inst.num_in_steps
        self.num_out_steps = ds_creator_inst.num_out_steps

        return None

    def LSTM_one_shot_predictor_named_i(self, ds_creator_inst, lstm_layer_size, dense_layer_size, n_LSTM_layers, n_dense_layers, extra_features):
        
        in_xy = Input(shape=(ds_creator_inst.num_in_steps, ds_creator_inst.num_in_features), name="in_xy")
        
        all_destinations = Input(shape=(8,3), name="all_destinations")
        n = Flatten()(all_destinations)

        m = LSTM(lstm_layer_size, return_sequences=True)(in_xy)
        
        for i in range(n_LSTM_layers - 1):
            m = LSTM(lstm_layer_size, return_sequences=False)(m)

        m = Flatten()(m)
        m = Concatenate()([m, n])

        m = Dense(dense_layer_size)(m)
        for i in range(max(n_dense_layers - 2, 0)):
            m = Dense(dense_layer_size)(m)
        
        m = Dense(ds_creator_inst.num_out_steps*ds_creator_inst.num_out_features,
                                    kernel_initializer=tf.initializers.random_uniform)(m)
        m = Reshape([ds_creator_inst.num_out_steps, ds_creator_inst.num_out_features])(m)

        model = tf.keras.Model(
            [in_xy, all_destinations],
            [m]
        )
        print(model.summary())
        self.model = model        

        # copy the scaler
        self.scaler = ds_creator_inst.normer

        # set some helper vars
        self.num_in_features = ds_creator_inst.num_in_features
        self.num_out_features = ds_creator_inst.num_out_features

        self.num_in_steps = ds_creator_inst.num_in_steps
        self.num_out_steps = ds_creator_inst.num_out_steps

        return None

    def __PermaDropout(self, rate):
        return Lambda(lambda x: K.dropout(x, level=rate))

    def LSTM_one_shot_predictor_epi(self, ds_creator_inst, dropout_rate, lstm_layer_size, dense_layer_size, n_LSTM_layers, n_dense_layers):
        in_xy = Input(shape=(ds_creator_inst.num_in_steps, ds_creator_inst.num_in_features), name="in_xy")
        m = LSTM(lstm_layer_size, return_sequences=True)(in_xy)
        
        for i in range(n_LSTM_layers - 1):
            m = LSTM(lstm_layer_size, return_sequences=False)(m)

        for i in range(n_dense_layers):
            m = Dense(dense_layer_size)(m)
            m = self.__PermaDropout(dropout_rate)(m)
        
        m = Dense(ds_creator_inst.num_out_steps*ds_creator_inst.num_out_features,
                                    kernel_initializer=tf.initializers.random_uniform)(m)
        m = Reshape([ds_creator_inst.num_out_steps, ds_creator_inst.num_out_features])(m)

        model = tf.keras.Model(
            [in_xy],
            [m]
        )

        self.model = model

        # copy the scaler
        self.scaler = ds_creator_inst.normer

        # set some helper vars
        self.num_in_features = ds_creator_inst.num_in_features
        self.num_out_features = ds_creator_inst.num_out_features

        self.num_in_steps = ds_creator_inst.num_in_steps
        self.num_out_steps = ds_creator_inst.num_out_steps

        return None
    


    # Setters/Getters
    @property
    def max_epochs(self):
        return self._max_epochs
    @max_epochs.setter  
    def max_epochs(self, max_epochs):
        self._max_epochs = max_epochs

    @property
    def patience(self):
        return self._patience
    @patience.setter
    def patience(self, patience):
        self._patience = patience

    @property
    def loss_function(self):
        return self._loss_function
    @loss_function.setter
    def loss_function(self, loss_function):
        self._loss_function = loss_function

    @property
    def optimizer(self):
        return self._patience
    @optimizer.setter
    def patience(self, optimizer):
        self._optimizer = optimizer

    @property
    def metric(self):
        return self._metric
    @patience.setter
    def patience(self, patience):
        self._patience = patience

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, model):
        self._model = model

    @property
    def scaler(self):
        return self._scaler
    @scaler.setter
    def scaler(self, scaler):
        self._scaler = scaler

    @property
    def num_in_steps(self):
        return self._num_in_steps
    @num_in_steps.setter
    def num_in_steps(self, value):
        self._num_in_steps = value

    @property
    def num_out_steps(self):
        return self._num_out_steps
    @num_out_steps.setter
    def num_out_steps(self, value):
        self._num_out_steps = value

    @property
    def num_in_features(self):
        return self._num_in_features
    @num_in_features.setter
    def num_in_features(self, value):
        self._num_in_features = value

    @property
    def num_out_features(self):
        return self._num_out_features
    @num_out_features.setter 
    def num_out_features(self, value):
        self._num_out_features = value

    @property
    def max_epochs(self):
        return self._max_epochs
    @max_epochs.setter
    def max_epochs(self, value):
        self._max_epochs = value
