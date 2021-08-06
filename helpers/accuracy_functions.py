from logging import error
from os import remove
from numpy.lib.function_base import average
import tensorflow as tf
from scipy.spatial import distance
import numpy as np
from math import sqrt

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

def avg_displacement_error(array1, array2, n, in_length_perc=1.):
  ''' Get the Average Displacement Error n steps into the future, n is 0 based '''  
  # remove the padding vals  
  array1 = remove_padding_vals(array1)
  array2 = remove_padding_vals(array2)

  # get the max number of points in the arrays
  min_len = min([len(array1), len(array2)])

  count = 0
  error_sum = 0.

  if min_len > n:
    def calc_distances(a):
      return np.sqrt(np.square(a[0]-a[2])+np.square(a[1]-a[3]))

    comb = np.hstack([array1[:(n+1)],array2[:(n+1)]])

    ade = np.average(np.apply_along_axis(calc_distances, 1, comb))

    # for x1, y1, x2, y2 in zip(array1[:,0], array1[:,1], array2[:,0], array2[:,1]):
    #   error_sum += sqrt((x1-x2)**2+(y1-y2)**2)
    #   count+=1

    #   ade = error_sum/count  
  else:
    ade = None
  
  
  return ade

def final_displacement_error(array1, array2, n):
  ''' Get the Final Displacement Error n steps into the future, n is 0 based '''
  array1 = remove_padding_vals(array1)
  array2 = remove_padding_vals(array2)

  if len(array1) > n and len(array2) > n:
    # actually faster than np.linalg.norm(a-b)
    fde = sqrt((array1[n][0]-array2[n][0])**2+(array1[n][1]-array2[n][1])**2)
  else:
    fde = None
  return fde

def remove_padding_vals2(array, padding_val = 0.):
    ''' remove the padding zeros from an array '''
    array_c = np.array(array)
    i = 0
    for i in range(len(array_c) - 1, 0, -1):
        if not (array_c[i,0]==padding_val and array_c[i,1]==padding_val):
            break    

    return array_c[:i+1]

def remove_padding_vals(array, padding_array = np.array([0., 0.])):
  ''' remove the padding zeros from an array '''
  try:
    min_index = np.min((array==padding_array).all(axis=1).nonzero()[0])
    return array[:min_index]
  except:
    return array

  
def batch_kpi(kpi_function, batch1, batch2, n):
    counter = 0
    not_counter = 0
    kpi_sum = 0.
    for array1, array2 in zip(batch1, batch2):
        kpi_val = kpi_function(array1, array2, n)
        if kpi_val is not None:
          kpi_sum += kpi_val
          counter += 1
        else:
          not_counter += 1
    
    if counter > 0:
      return kpi_sum / counter, counter, not_counter
    else:
      return None, counter, not_counter
    
def tf_ds_kpi(tf_ds, keyword1, dl_trainer, n, var_in_len):
  counter = 0
  not_counter = 0
  ade_sum = 0.
  fde_sum = 0.

  iterable = iter(tf_ds)
  for d, o in iterable:
    batch1 = d[keyword1]

    batch2 = dl_trainer.predict_repetitively_dict(d, False, n+1, var_in_len)
    
    ade_val, c, nc = batch_kpi(avg_displacement_error, batch1, batch2, n)
    fde_val, _, _ = batch_kpi(final_displacement_error, batch1, batch2, n)

    not_counter += nc

    if ade_val is not None:
      ade_sum += ade_val * c
      counter += c    
    if fde_val is not None:
      fde_sum += fde_val * c

  if counter > 0:
    return ade_sum / counter, fde_sum / counter, counter, not_counter
  else:
    return None, None, counter, not_counter


def test():
    z = np.zeros((5,3,2))
    v = np.ones((5,2,2))
    tog = np.concatenate([v,z], axis=1)
    z2 = np.zeros((5,4,2))
    v2 = np.ones((5,3,2))
    tog2 = np.concatenate([v2,z2], axis=1)
    print(tog)

    a = np.array([[1., 1.], [2., 2.], [3., 3.], [0., 0.]])
    b = a + 1; b[:,1] -= 1

    print(final_displacement_error(a,b,2))
    print(avg_displacement_error(a,b,2))

    print(avg_displacement_error(a,b,3))

    aa = np.array([a,a,a])
    bb = np.array([b,b,[[1., 1.], [0., 0.], [0., 0.], [0., 0.]]])

    print(batch_kpi(avg_displacement_error, aa, bb, 2))

    d = {"a":aa, "b":bb}
    tf_ds=tf.data.Dataset.from_tensors(d)

    # Check if the new removal of paddings vals is faster
    # import timeit
    # c = np.zeros(shape=(230,2))
    # c = np.vstack([np.array([1., 0.]), c])
    # def fun():
    #   remove_padding_vals(c)
    # def fun2():
    #   remove_padding_vals2(c)
    # print(timeit.timeit(fun))
    # print(timeit.timeit(fun2))

    return None
if __name__ == '__main__':
  test()