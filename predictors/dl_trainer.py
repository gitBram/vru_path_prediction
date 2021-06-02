''' class for training deep learning models and setting parameters for the training phase'''

import tensorflow as tf

class DLTrainer:
    def __init__(self, max_epochs = 50, patience = 2, loss_function = tf.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(), metric = tf.metrics.MeanAbsoluteError()):
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
        self.model.load_weights(save_path)

    def predict(self, input_tensor, scale_input_tensor):
        # check that input is 3D (batch - time - feature)
        if len(input_tensor.shape) <= 2:
            input_tensor = tf.reshape(input_tensor, [1, -1, self.num_in_features])
        
        if scale_input_tensor:
            output = self.model(self.scaler.scale_tensor(input_tensor, "normalize", "in"))
        else:
            output = self.model(input_tensor)

        # scale if needed
        if self.scaler is not None:
            output = self.scaler.scale_tensor(output, "denormalize", "out")

        return output
    
    def predict_repetitively(self, input_tensor, scale_input_tensor, num_repetitions, fixed_len_input):
        assembled_input = tf.cast(input_tensor, dtype=tf.float32)
        assembled_output = None
        for i in range(num_repetitions):
            # get one prediction
            new_output = self.predict(assembled_input, scale_input_tensor)

            try:
                assembled_output = tf.concat([assembled_output, new_output], axis=1)
            except:
                assembled_output = new_output

            # add the output to the new input, drop first of input in case it input should be kept constant length

            assembled_input = tf.concat([assembled_input, new_output], axis=1)

            # drop the n first rows if input has to stay same length
            out_len = len(new_output)
            assembled_input = assembled_input[out_len:]

        return assembled_output


    def LSTM_one_shot_predictor(self, ds_creator_inst):

        lstm_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(ds_creator_inst.num_in_steps, ds_creator_inst.num_in_features)),
            # Shape [batch, time, features] => [batch, lstm_units]
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(64, return_sequences=False),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(112),
            tf.keras.layers.Dense(ds_creator_inst.num_out_steps*ds_creator_inst.num_out_features,
                                kernel_initializer=tf.initializers.random_uniform),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([ds_creator_inst.num_out_steps, ds_creator_inst.num_out_features])
        ])

        self.model = lstm_model

        # copy the scaler
        self.scaler = ds_creator_inst.normer

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
