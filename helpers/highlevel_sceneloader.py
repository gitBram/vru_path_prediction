''' 
This file contains a high-level dataloader. 
Certain datasets can be loaded, background image and necessary parameters about the image to be plotted will be returned.
Rather meant as a file where parameters that need to be determined by hand are stored and easily loaded when this dataset is needed again.

GOAL:
- Get dataframe as output
- get picture as output
- plot all paths on image
- plot list of different x/y matrices in different colors on image (fixed colors)
  - Save or return ax
'''

import xml.etree.ElementTree as ET
import os, sys, yaml
import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil
import tensorflow as tf

class HighLevelSceneLoader():
  def __init__(self, img_bound_file_loc, dest_file_loc):
    self._traj_dataframe = None
    self.image_limits = None
    self._image = None
    self.img_bound_dict = self.__populate_img_bound_dict('ind', img_bound_file_loc)
    self._dest_file_loc = dest_file_loc

    self.dataset_name = None
    self.scene_name = None

    #save col names for this df
    self._df_split_col = None
    self._df_x_col = None
    self._df_y_col = None
    
  def __populate_dest_dict(self):
    return None


  def __tree_iter_to_dict(self, node, val_type):
    names = []
    values = []

    for child in node:
      names.append(child.tag)
      values.append(float(child.text))

    return dict(zip(names, values))


  def __populate_img_bound_dict(self, dataset_name, file_loc):
    # create dict
    img_bound_dict = dict()

    # read the file
    tree = ET.parse(file_loc)
    root = tree.getroot()

    # transfer the xml file to the dict in the class
    for dataset in root:
      ds_name = dataset.attrib['name']
      if ds_name == dataset_name:
        for key in dataset:
          key_name = key.attrib['name']
          for img_bound in key:          
            names = []
            vals = []

            for val in img_bound:
              print(val.tag)
              print(val.text)
              names.append(val.tag)
              vals.append(float(val.text))

            img_bound_dict[(ds_name, key_name)] = dict(zip(names, vals))    
    return img_bound_dict

  
  def load_ind(self, root_datasets, file_id, end_file_id_range=None,
  x_col = 'pos_x', y_col = 'pos_y', split_col = 'agent_id'):

      # set the col names correctly
      self.df_split_col = split_col
      self.df_x_col = x_col
      self.df_y_col = y_col

      # import the data importer  
      from OpenTraj.toolkit.loaders import loader_ind

      # import ind data
      ind_root = os.path.join(root_datasets, 'inD-dataset-v1.0/data')
      
      # see whether only one or multiple files need to be loaded
      if end_file_id_range is not None:
        # we want to load a range of files

        # get the first file in there
        i = file_id
        ind_dataset = loader_ind.load_ind(os.path.join(ind_root, '%02d_tracks.csv' % i),
                              scene_id='1-%02d' %i, sampling_rate=36, use_kalman=False).data
        i += 1
        while True:
          # now append the other wanted files, making sure the path id is correct          
          add_file_data = loader_ind.load_ind(os.path.join(ind_root, '%02d_tracks.csv' % i),
                                scene_id='1-%02d' %i, sampling_rate=36, use_kalman=False).data
          
          max_id = ind_dataset[split_col].max()
          add_file_data[split_col] += max_id
          ind_dataset = ind_dataset.append(add_file_data)     

          print("Index %i and length %i"%(i, len(ind_dataset)))     

          i += 1
          if i > end_file_id_range:
            break
      else:
        # we only want to load one file
        ind_dataset = loader_ind.load_ind(os.path.join(ind_root, '%02d_tracks.csv' % file_id),
                                scene_id='1-%02d' %file_id, sampling_rate=36, use_kalman=False).data
      
      im = plt.imread(os.path.join(ind_root, '%02d_background.png'%(file_id)))

      # set the easily accessable class vals
      self._traj_dataframe = ind_dataset
      self._image = im
      self.image_limits = self.img_bound_dict[('ind', str(file_id))]
      self.dataset_name = 'ind'
      self.scene_name = str(file_id)

  def load_sdd(self, opentraj_root, scene_name, scene_video_id, 
  x_col = 'pos_x', y_col = 'pos_y', split_col = 'agent_id'):

      # set the col names correctly
      self.df_split_col = split_col
      self.df_x_col = x_col
      self.df_y_col = y_col

      # import the data importer  
      from toolkit.loaders.loader_sdd import load_sdd, load_sdd_dir

      # fixme: replace OPENTRAJ_ROOT with the address to root folder of OpenTraj
      sdd_root = os.path.join(opentraj_root, 'datasets', 'SDD')
      annot_file = os.path.join(sdd_root, scene_name, scene_video_id, 'annotations.txt')
      image_file = plt.imread(os.path.join(sdd_root, scene_name, scene_video_id, 'reference.jpg'))

      # load the homography values
      with open(os.path.join(sdd_root, 'estimated_scales.yaml'), 'r') as hf:
          scales_yaml_content = yaml.load(hf, Loader=yaml.FullLoader)
      scale = scales_yaml_content[scene_name][scene_video_id]['scale']

      ind_dataset = load_sdd(annot_file, scale=scale, scene_id=scene_name + '-' + scene_video_id,
                              drop_lost_frames=False,sampling_rate=36, use_kalman=False) 

      df_ind_trajectories = ind_dataset.data
      # only get pedestrians
      df_ind_trajectories=df_ind_trajectories[df_ind_trajectories['label']=='pedestrian']

      # set the easily accessable class vals
      self._traj_dataframe = df_ind_trajectories
      self._image = image_file
      # self.image_limits = self.img_bound_dict[('sdd', str(scene_name)+str(scene_video_id))]
      self.dataset_name = 'sdd'
      self.scene_name = str(scene_name) + str(scene_video_id)
  
  def remove_label_fill_vals(self, input_tensor, fill_val = 0.):
    '''
    Since tensorflow did now allow to concat datasets mixing RaggedTensor and Tensor, here we go
    '''
    mask = tf.equal(input_tensor, tf.fill([1,2], fill_val))

    return tf.boolean_mask(input_tensor, mask)


  def plot_all_paths(self, ms = 3, save_path = None, ax=None, hide_axes=False):
      paths = list(tuple(self.traj_dataframe.groupby(self.df_split_col)))
      lst_realxy_mats = [path[1][[self.df_x_col, self.df_y_col]].to_numpy() for path in paths]
      self.plot_on_image(lst_realxy_mats, ms = 3, invert_y = False, save_path = save_path, 
      ax=ax, col_num_dicts=dict(zip(["x", "y"], [0, 1])), labels=None,
      axes_labels=["x position [m]", "y position [m]"], hide_axes=hide_axes)


  def plot_on_image(self, lst_realxy_mats, ms = 3, invert_y = False, save_path = None, 
  ax=None, col_num_dicts=dict(zip(["x", "y"], [0, 1])), labels=None, colors=None, title=None, 
  axes_labels=None, hide_axes=False):
    ''' Plot a list of xy matrices on top of an image '''
    # Check input types
    if type(col_num_dicts)!= list:
      col_num_dicts = [col_num_dicts]

    # try to expand the dicts list to correct shape by copying it
    if len(col_num_dicts) == 1 and len(lst_realxy_mats) > 1:
      col_num_dicts = col_num_dicts * len(lst_realxy_mats)
    
    # Sanity check
    if len(lst_realxy_mats) != len(col_num_dicts):
      raise ValueError("Number of xy matrices and dictionaries should be equal.")

    # Set up axis 
    if ax is None:
        ax = plt.gca()
    else:
      plt.sca(ax)
    
    ax.set_aspect('equal', adjustable='box')
    extent_bounds = [self.image_limits['x_min'], self.image_limits['x_max'],
    self.image_limits['y_min'], self.image_limits['y_max']]
    ax.imshow(self.image, extent=list(extent_bounds))

    # in-built possibility to reverse y-axis
    if invert_y:
      try:
        for a in ax:
          a.invert_yaxis()
      except:
        ax.invert_yaxis()

    for xy, i, col_num_dict in zip(lst_realxy_mats, range(len(lst_realxy_mats)), col_num_dicts):
   
      # Plot if there is actual data
      if np.size(xy) > 0:
        # start by removing zeros, if any
        xy_np = np.array(xy)        
        # xy_np = np.array(self.remove_label_fill_vals(xy_np))
        my_shape = xy_np.shape

        # Check that data is not 1d, can be the case if only one point is fed and matrix is squeezed
        if len(my_shape) == 1:
          xy_np = xy_np.reshape(1,-1)

        # Reshape 3d (batched) data for easy plotting
        if len(my_shape) == 3:
          xy_np = xy_np.reshape(-1, my_shape[-1])         

        # get the correct marker size
        m_size = None
        if type(ms) == list:
          m_size = ms[i]
        else:
          m_size = ms
        ax.scatter(xy_np[:, col_num_dict['x']], xy_np[:, col_num_dict['y']], s=m_size, c=colors)

    if not labels is None:
      ax.legend(labels)

    if not axes_labels is None:
      ax.set_xlabel(axes_labels[0])
      ax.set_ylabel(axes_labels[1])
    
    if not title is None:
      plt.gcf().suptitle(title)    

    if hide_axes:
      plt.axis("off")

    if save_path is not None:
      plt.gcf()
      plt.savefig(save_path, bbox_inches='tight')
    return None

  def add_circles(self, centres_mat, radius, ax = None, save_path = None, color='b'):
    if ax is None:
        ax = plt.gca()
    else:
      plt.sca(ax)

    for centre in centres_mat:
      circle = plt.Circle((centre[0], centre[1]), radius, color=color, fill=False)
      ax.add_patch(circle)
    if save_path is not None:
      plt.gcf()
      plt.savefig(save_path, dpi=500)
    return None

  def plot_dest_probs(self, dest_locs, dest_probs, min_marker_size, max_marker_size, ax = None, save_path = None):
    ''' for plotting the destination probabilities, visible by the size of the probability '''
    # assert numpy format
    dest_locs_mat = np.array((dest_locs))
    dest_probs_mat = np.array((dest_probs))
    # basic checks
    assert len(dest_locs_mat) == len(dest_probs_mat) 
    assert max_marker_size > min_marker_size 



    # pyplot things
    if ax is None:
        ax = plt.gca()
    else:
      plt.sca(ax)
      
    # figure out how to scale probs to marker size
    probs_min = np.nanmin(dest_probs_mat)
    probs_max = np.nanmax(dest_probs_mat)
    p_s_scalef = (max_marker_size-min_marker_size)/(probs_max - probs_min)

    # construct list of marker sizes for each point    
    num_locs = len(dest_locs_mat)
    sizes = []
    for i in range(num_locs):
      p = dest_probs[i]

      # put in try as NaN is possible ofr unreachable dests
      try:
        size = floor(min_marker_size + p_s_scalef * p)
        sizes.append(size)
      except:
        sizes.append(min_marker_size)
    
    # plot on the figure
    ax.scatter(dest_locs_mat[:, 0], dest_locs_mat[:, 1], s=sizes)

    # save if needed
    if save_path is not None:
      plt.gcf()
      plt.savefig(save_path)

    return None

  def df_to_lst_realxy_mats(self):
    ''' Convert a dataframe to a list of xy matrices based on a value in split_col '''
    df = self.traj_dataframe.copy()
    sc = self.df_split_col
    split_ids = df[sc].unique()
    out_xy = []
    for split_id in split_ids:
      out_xy.append(df[df[self.df_split_col]==split_id][[self.df_x_col, self.df_y_col]].to_numpy())
    return out_xy     

  def df_to_lst_realxy_mats_ext(self, df):
    ''' Convert a dataframe to a list of xy matrices based on a value in split_col '''
    df = df.copy()
    sc = self.df_split_col
    split_ids = df[sc].unique()
    out_xy = []
    for split_id in split_ids:
      out_xy.append(df[df[self.df_split_col]==split_id][[self.df_x_col, self.df_y_col]].to_numpy())
    return out_xy   

  def plot_all_trajs_on_img(self, save_path, col_num_dicts=dict(zip(["x", "y"], [0, 1]))):
    ''' Plot all the trajectories on the background image '''
    l = self.df_to_lst_realxy_mats()
    ax = self.plot_on_image(l, save_path=save_path, invert_y=False, col_num_dicts=col_num_dicts)
    return ax

  def plot_dests_on_img(self, save_path, col_num_dicts=dict(zip(["x", "y"], [0, 1]))):
    ''' Plot the retrieved destinations on the background image ''' 
    d = self.destination_matrix
    ax = self.plot_on_image([d], save_path=save_path, invert_y=False, col_num_dicts=col_num_dicts)
    return ax

  @property
  def df_split_col(self):
    return self._df_split_col
  @df_split_col.setter
  def df_split_col(self, data):
    self._df_split_col = data

  @property
  def df_x_col(self):
    return self._df_x_col    
  @df_x_col.setter
  def df_x_col(self, data):
    self._df_x_col = data

  @property
  def df_y_col(self):
    return self._df_y_col    
  @df_y_col.setter
  def df_y_col(self, data):
    self._df_y_col = data

  @property
  def image(self):
    return self._image
  @image.setter
  def image(self, im):
    self._image = im

  @property
  def traj_dataframe(self):
    return self._traj_dataframe
  
  @traj_dataframe.setter
  def traj_dataframe(self, df):
    ''' correctly set the df, with resetting index '''
    self._traj_dataframe = df.reset_index()

  @property
  def destination_matrix(self):
    # read the file
    tree = ET.parse(self._dest_file_loc)
    root = tree.getroot()

    # get the dataset branch
    ds_branch = root.find('dataset/[@name="%s"]'%(self.dataset_name))

    # get key branch in the branch
    key_branch = ds_branch.find('key/[@name="%s"]'%(str(self.scene_name)))

    # find all locations for this key
    locs = list(key_branch.findall("./dest"))
    locs_mat = np.zeros((len(locs), 2))
    for loc, i in zip(locs, range(len(locs))):
      d = self.__tree_iter_to_dict(loc, 'not_used')
      locs_mat[i, 0] = d['x']
      locs_mat[i, 1] = d['y']
    
    return locs_mat

"""
def plot_on_image(image, real_boundaries, lst_realxy_mats, ms = 3, invert_y = False, ax = None):
  ''' Plot a list of xy matrices on top of an image '''
  if ax is None:
    fig, ax = plt.subplots()
  ax.set_aspect('equal', adjustable='box')
  ax.imshow(image, extent=real_boundaries)

  if invert_y:
    try:
      for a in ax:
        a.invert_yaxis()
    except:
      ax.invert_yaxis()
  for xy in lst_realxy_mats:
    ax.scatter(xy[:, 0], xy[:, 1], s=ms)

  plt.savefig('hello.png')
  return ax
  """

def __test():

  curr_p = os.getcwd()
  opentraj_root = os.path.join(curr_p, 'OpenTraj')
  sys.path.append(opentraj_root)
  sys.path.append(curr_p)
  
  rel_p_img_b = 'helpers/analysed_vars_storage/img_bounds.xml'
  rel_p_dests = 'helpers/analysed_vars_storage/destination_locations.xml'
  p_img_bounds = os.path.join(curr_p, rel_p_img_b)
  p_dest_locs = os.path.join(curr_p, rel_p_dests)
  a = HighLevelSceneLoader(p_img_bounds,p_dest_locs)
  p_to_data = os.path.join(curr_p, 'data/path_data')
  a.load_ind(p_to_data,11)


if __name__ == '__main__':
    __test()