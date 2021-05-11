''' module to find the waypoints '''

#TODO: start off with a grid of booleans that state for each cell whether value is updated
# interest_point_searcher should only updated necessary cells upon calling 


from sklearn.cluster import KMeans
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import cv2 as cv

class WaypointAnalyser:
    def __init__(self, df_data, grid_res, grid_limits, look_around_dist, path_id_col = 'agent_id', x_col = 'pos_x', y_col = 'pos_y', index_col = 'index',
                            n_clusters = 2):
        self.df_data = df_data.reset_index(drop=True)
        self.grid_res = grid_res
        self.grid_limits = grid_limits
        self.look_around_dist = look_around_dist
        self.path_id_col = path_id_col
        self.x_col = x_col
        self.y_col = y_col
        self.index_col = index_col
        self.n_clusters = n_clusters

        # interest_areas = self.interest_area_searcher(save_img = True)
        # interest_points = self.interest_point_searcher(interest_areas, save_img = True)

    def interest_area_searcher(self, save_img = False):
        ''' find for each grid cell a measure for being a point of interest '''
        g_x_min, g_x_max, g_y_min, g_y_max = self.grid_limits
        n_x_cells, n_y_cells = math.ceil((g_x_max-g_x_min)/self.grid_res), math.ceil((g_y_max-g_y_min)/self.grid_res)

        value_matrix = np.zeros(shape=(n_y_cells, n_x_cells))
        for c_x_ind in range(n_x_cells):
            for c_y_ind in range(n_y_cells):
                cell_lims = self.get_cell_limits((c_x_ind, c_y_ind), self.grid_limits, self.grid_res)
                points_matrix = self.extract_neighbour_points(cell_lims)
                # cluster
                clusterlist = []

                try:
                    if self.n_clusters > 1:
                        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(points_matrix)
                        for i in range(self.n_clusters):
                            clusterlist.append(points_matrix[kmeans.labels_ == i])
                    else:
                        clusterlist.append(points_matrix)
                except: # no points for this one
                    clusterlist = [np.matrix([0.])] 
                # calculate value for each grid cell
                value = self._value_function(clusterlist)
                # put value in correct place in matrix
                value_matrix[c_y_ind, c_x_ind] = value #value_matrix[(n_y_cells-1) - c_y_ind, c_x_ind] = value

        #TODO: get dirty fix below out
        value_matrix = np.nan_to_num(value_matrix)
        print(value_matrix)
        print(np.max(value_matrix))

        if save_img:
            self._save_2d_matrix_img(value_matrix, 'data/images/interest_areas.jpg', invert_y=True)
                
        return value_matrix

    def interest_point_searcher(self, interest_area_matrix, save_img=False, kernel_size = 3, min_dist = 6, thresh_rel = .1):

        # blur matrix to smooth out information locally
        kernel = np.ones((3,3),np.float32)/16
        out_vals_blurred = cv.filter2D(interest_area_matrix,-1,kernel)
        out_vals_blurred_twice = cv.filter2D(out_vals_blurred,-1,kernel)

        if save_img:
            self._save_2d_matrix_img(out_vals_blurred_twice, 'data/images/interest_areas_bl.jpg', invert_y=True)

        
        # apply Laplacian to get stronger peak effect
        ddepth = cv.CV_64F 
        lapl = cv.Laplacian(out_vals_blurred_twice, ddepth, ksize=kernel_size)
        lapl_blurred = cv.filter2D(lapl,-1,kernel)

        if save_img:
            self._save_2d_matrix_img(lapl_blurred, 'data/images/laplacian.jpg', invert_y=True)

        # find local peaks on laplacian to export as waypoints
        coordinates_wayp_bl = self._switch_columns(peak_local_max(-lapl_blurred, min_distance=min_dist, threshold_rel=thresh_rel))

        if save_img:    
            self._save_2d_matrix_scatter_img(lapl_blurred, coordinates_wayp_bl, 'data/images/laplacian.jpg', invert_y=True)

        return coordinates_wayp_bl

    def _switch_columns(self, mat):
        ''' switch the 2 columns of a 2d matrix '''

        return mat[:,[1, 0]]

    def _value_function(self, matrix_list): 
        ''' Function to convert extracted cluster data to decimal representing likelihood of being a waypoint '''
        value = 0
        for matrix in matrix_list:
            std = matrix.std(axis=0)
            sq = np.square(std)
            value = value + np.sum(sq)
        return value

    def extract_neighbour_points(self, cell_limits):
        ''' For cell in the grid, check which paths go through cell and extract points n steps before and after for cluster analysis '''
        out_df = self.df_data.copy()
        # filter by cel limits
        c_x_min, c_x_max, c_y_min, c_y_max = cell_limits
        filter_expression = (out_df[self.x_col] >= c_x_min) & (out_df[self.x_col] < c_x_max) & (out_df[self.y_col] >= c_y_min) & (out_df[self.y_col] < c_y_max)
        loc_filter_df = out_df[filter_expression]

        # TODO: probably good idea to filter out if multiple entries for one path exist in this cell

        # find path points before and after
        path_ids = loc_filter_df[self.path_id_col].unique().tolist()

        points = []
        for path_id in path_ids:
            # get index of this id
            row_current = loc_filter_df[loc_filter_df[self.path_id_col]==path_id]
            row_current_path_id = row_current[self.path_id_col].values[0]
            row_current_ind = row_current.index[0]
            # get index of n steps before and after
            prev_ind = row_current_ind - self.look_around_dist    
            next_ind = row_current_ind + self.look_around_dist
            # bound the index to the available index range
            indices = out_df.index
            min_index = min(indices)
            max_index = max(indices)
            prev_ind = self._cut_off_indices(prev_ind, min_index, max_index)
            next_ind = self._cut_off_indices(next_ind, min_index, max_index)
            # get rows n steps before and after
            row_prev = out_df.iloc[prev_ind,:]
            row_next = out_df.iloc[next_ind,:]
            # get path x and y for indices after checking whether they belong to same path
            row_prev_path_id = row_prev[self.path_id_col] # get path id
            row_next_path_id = row_next[self.path_id_col] # get path id
            if row_prev_path_id == path_id:
                this_x = row_prev[self.x_col] - row_current[self.x_col].values[0]
                this_y = row_prev[self.y_col] - row_current[self.y_col].values[0]
                points.append([this_x, this_y])
            if row_next_path_id == path_id:
                this_x = row_next[self.x_col] - row_current[self.x_col].values[0]
                this_y = row_next[self.y_col] - row_current[self.y_col].values[0]
                points.append([this_x, this_y])

        return np.matrix(points)

    def _cut_off_indices(self, index, min_index, max_index):
        ''' to make sure not to go out of df bounds '''
        if index < min_index:
            return min_index
        elif index > max_index:
            return max_index
        else:
            return index

    def get_cell_limits(self, xy, grid_limits, grid_resolution):
        ''' get the x/y limits of a cell with coordinates (x,y), in order to filter the df on it '''
        # xy is tuple with cell x/y index [no unit], grid_limits is tuple (xmin, xmax, ymin, ymax) [m], grid_resolution is width of cell [m]
        # based on grid definition
        # grid zero is bottom left, zero based indexing

        # represent input easier
        x,y = xy
        g_x_min, g_x_max, g_y_min, g_y_max = grid_limits
        # check input
        x_diff = g_x_max - g_x_min
        y_diff = g_y_max - g_y_min

        assert x * grid_resolution < x_diff
        assert y * grid_resolution < y_diff

        # do calculations
        x_min = g_x_min + grid_resolution * x
        x_max = x_min + grid_resolution
        y_min = g_y_min + grid_resolution * y
        y_max = y_min + grid_resolution

        return x_min, x_max, y_min, y_max

    def _save_2d_matrix_img(self, matrix, location, invert_y=True):
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.imshow(matrix, cmap='cool', interpolation='nearest')
        if invert_y:
            ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        fig.savefig(location)

    def _save_2d_matrix_scatter_img(self, matrix, scatterpoints, location, invert_y=True):
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.imshow(matrix, cmap='cool', interpolation='nearest')
        ax.scatter(scatterpoints[:,0], scatterpoints[:,1], c='yellow')
        if invert_y:
            ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        fig.savefig(location)


# Test function for module  
def _test():
    return None

if __name__ == '__main__':
    _test()
