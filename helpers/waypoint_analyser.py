from sklearn.cluster import KMeans
import pandas as pd

class WaypointAnalyser:
    def __init__(df_data, grid_res, grid_limits, look_around_dist, path_id_col = 'agent_id', x_col = 'pos_x', y_col = 'pos_y', index_col = 'index',
                            n_clusters = 2):
        self.df_data = df_data.reset_index()
        self.grid_res = grid_res
        self.grid_limits = grid_limits
        self.look_around_dist = look_around_dist
        self.path_id_col = path_id_col
        self.x_col = x_col
        self.y_col = y_col
        self.index_col = index_col

    def interest_point_searcher(look_around_dist, valueFunction):
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
                    if n_clusters > 1:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points_matrix)
                        for i in range(n_clusters):
                            clusterlist.append(points_matrix[kmeans.labels_ == i])
                    else:
                        clusterlist.append(points_matrix)
                except: # no points for this one
                    clusterlist = [np.matrix([0.])] 
                # calculate value for each grid cell
                value = self._value_function(clusterlist)
                # put value in correct place in matrix
                value_matrix[(n_y_cells-1) - c_y_ind, c_x_ind] = value #value_matrix[(n_y_cells-1) - c_y_ind, c_x_ind] = value
                
        return value_matrix

    def _value_function(matrix_list): 
        ''' Function to convert extracted cluster data to decimal representing likelihood of being a waypoint '''
        value = 0
        for matrix in matrix_list:
            std = matrix.std(axis=0)
            sq = np.square(std)
            value = value + np.sum(sq)
        return value

    def extract_neighbour_points(cell_limits):
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
            prev_ind = row_current_ind - look_around_dist    
            next_ind = row_current_ind + look_around_dist
            # bound the index to the available index range
            indices = out_df.index
            min_index = min(indices)
            max_index = max(indices)
            prev_ind = self.cut_off_indices(prev_ind, min_index, max_index)
            next_ind = self.cut_off_indices(next_ind, min_index, max_index)
            # get rows n steps before and after
            print(prev_ind)
            row_prev = out_df.iloc[prev_ind,:]
            row_next = out_df.iloc[next_ind,:]
            # get path x and y for indices after checking whether they belong to same path
            row_prev_path_id = row_prev[self.path_id_col] # get path id
            row_next_path_id = row_next[self.path_id_col] # get path id
            if row_prev_path_id == path_id:
                this_x = row_prev[self.x_col] - row_current[self.x_col].values[0]
                this_y = row_prev[y_col] - row_current[y_col].values[0]
                points.append([this_x, this_y])
            if row_next_path_id == path_id:
                this_x = row_next[self.x_col] - row_current[self.x_col].values[0]
                this_y = row_next[y_col] - row_current[y_col].values[0]
                points.append([this_x, this_y])

        return np.matrix(points)

    def cut_off_indices(index, min_index, max_index):
        ''' to make sure not to go out of df bounds '''
        if index < min_index:
            return min_index
        elif index > max_index:
            return max_index
        else:
            return index

    def get_cell_limits(xy, grid_limits, grid_resolution):
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

# Test function for module  
def _test():
    return None

if __name__ == '__main__':
    _test()
