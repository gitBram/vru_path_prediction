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
import os, sys
import matplotlib.pyplot as plt

class HighLevelSceneLoader():
  def __init__(self, dataset_name, img_bound_loc):
    self._traj_dataframe = None
    self.image_limits = None
    self._image = None
    self.img_bound_dict = self.__populate_img_bound_dict('ind', img_bound_loc)


  def __populate_img_bound_dict(self, dataset_name, source_loc):
    # create dict
    img_bound_dict = dict()

    # read the file
    tree = ET.parse(source_loc)
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
              names.append(val.tag)
              vals.append(float(val.text))

            img_bound_dict[(ds_name, key_name)] = dict(zip(names, vals))    
    return img_bound_dict

  
  def load_ind(self, root_datasets, file_id):

      from OpenTraj.toolkit.loaders import loader_ind

      # import ind data
      ind_root = os.path.join(root_datasets, 'inD-dataset-v1.0/data')
      ind_dataset = loader_ind.load_ind(os.path.join(ind_root, '%02d_tracks.csv' % file_id),
                              scene_id='1-%02d' %file_id, sampling_rate=36, use_kalman=False)
      im = plt.imread(os.path.join(ind_root, '%02d_background.png'%(file_id)))

      # set the easily accessable class vals
      self._traj_dataframe = ind_dataset.data
      self._image = im
      self.image_limits = self.img_bound_dict[('ind', str(file_id))]

  def load_sdd(self, opentraj_root, scene_name, scene_video_id):
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
      self.image_limits = self.img_bound_dict[('sdd', str(scene_name)+str(scene_video_id))]

  # def __retrieve_img_bounds(self, dataset, scene_id):
  #   ''' in order to accurately display dataset on the image, get the physical boundaries of the background image '''
  #   return self.img_bound_dict[(dataset, scene_id)]

  def plot_on_image(self, lst_realxy_mats, ms = 3, invert_y = False, save_path = None, ax = None):
    ''' Plot a list of xy matrices on top of an image '''
    if ax is None:
      fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.imshow(self.image, extent=list(self.image_limits.values()))

    if invert_y:
      try:
        for a in ax:
          a.invert_yaxis()
      except:
        ax.invert_yaxis()
    for xy in lst_realxy_mats:
      ax.scatter(xy[:, 0], xy[:, 1], s=ms)

    if save_path is not None:
      plt.savefig(save_path)
    return ax

  def df_to_lst_realxy_mats(self, df, x_col = 'pos_x', y_col = 'pos_y', split_col = 'agent_id'):
    ''' Convert a dataframe to a list of xy matrices based on a value in split_col '''
    split_ids = df[split_col].unique()
    out_xy = []
    for split_id in split_ids:
      out_xy.append(df[df[split_col]==split_id][[x_col, y_col]].to_numpy())
    return out_xy        

  def plot_all_trajs_on_img(self, save_path):
    l = self.df_to_lst_realxy_mats(self.traj_dataframe)
    ax = self.plot_on_image(l, save_path=save_path)
    return ax


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
    self._traj_dataframe = df.reset_index()
    



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

def __test():

  curr_p = os.getcwd()
  opentraj_root = os.path.join(curr_p, 'OpenTraj')
  sys.path.append(opentraj_root)
  sys.path.append(curr_p)
  
  rel_p = 'helpers/analysed_vars_storage/img_bounds.xml'
  p = os.path.join(curr_p, rel_p)
  a = HighLevelSceneLoader('ind',p)
  p_to_data = os.path.join(curr_p, 'data/path_data')
  a.load_ind(p_to_data,11)
  print(a.img_bound_dict)
  a.plot_all_trajs_on_img('well.png')

if __name__ == '__main__':
    __test()