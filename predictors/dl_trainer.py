''' class for training deep learning models and setting parameters for the training phase'''

from _typeshed import OpenTextModeWriting
import tensorflow as tf

class DLTrainer:
  def __init__(self, max_epochs = 50, patience = 2, loss_function = tf.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(), metric = tf.metrics.MeanAbsoluteError()):
    self._max_epochs = max_epochs
    self._patience = patience
    self._loss_function = loss_function
    self._optimizer = optimizer
    self._metric = metric

  def compile_and_fit(self, model, dataset_dict):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=self._patience,
                                                      mode='min',
                                                      restore_best_weights=True)

    model.compile(loss=self._loss_function,
                  optimizer=self._optimizer,
                  metrics=[self._metric])
    cb = [early_stopping]

    history = model.fit(dataset_dict['train'], epochs=self._max_epochs,
                        validation_data=dataset_dict['val'],
                        callbacks=cb)
    return history

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
        self._metric = metric