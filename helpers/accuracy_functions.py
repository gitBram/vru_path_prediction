import tensorflow as tf
from scipy.spatial import distance
import numpy as np

def return_batch_accuracy(tensor_1, tensor_2, n=None):
    '''
    Get a value for accuracy, comparing 2 point arrays.
    If n is provided, only predictions where label of at least leng
    '''
    # get the shape right - we will be working on time dimension
    if tensor_1.ndim == 2:
      arr_1_c = [tensor_1]
    if tensor_2.ndim == 2:
      arr_2_c = [tensor_2]
    if tensor_1.ndim == 3:
      arr_1_c = tf.unstack(tensor_1)
    if tensor_2.ndim == 3:
      arr_2_c = tf.unstack(tensor_2)

    # remove zero fill values
    arr_2_c = [remove_padding_vals(x) for x in arr_2_c]
    arr_1_c = [remove_padding_vals(x) for x in arr_1_c]

    # get the shortest length for each pair
    distance_list = []
    number_of_points_included = []
    for a1, a2 in zip(arr_1_c, arr_2_c):  

      a1 = tf.squeeze(a1)
      a2 = tf.squeeze(a2)
      shortest_len = min(a1.shape[-2], a2.shape[-2])
      number_of_points_included.append(shortest_len)
      a1 = a1[:shortest_len, :]
      a2 = a2[:shortest_len, :]

      summed_dist = np.average(np.diag(distance.cdist(a1, a2, "euclidean")))
      distance_list.append(summed_dist)

    return distance_list, number_of_points_included

def remove_padding_vals(array, padding_val = 0.):
    ''' remove the padding zeros from an array '''
    array_c = np.array(array)
    i = 0
    for i in range(len(array_c) - 1, 0, -1):
        if not (array_c[i,0]==padding_val and array_c[i,1]==padding_val):
            break    

    return array_c[:i+1]

def return_ds_accuracy(model_trainer, dataset):
    return None

def test():
    z = np.zeros((5,2))
    v = np.random.random((6,2))
    tog = np.concatenate([v,z], axis=0)
    print(tog)
    print(remove_zeros(tog))
    return None
if __name__ == '__main__':
    test()