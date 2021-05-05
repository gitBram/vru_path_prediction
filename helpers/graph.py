import numpy as np
import warnings
from math import floor, ceil, sqrt

# TODO: Add observers in order to only do the necessary recalculations https://stackoverflow.com/questions/6190468/how-to-trigger-function-on-value-change

class Graph():
    ''' Graph class to fill transition matrix and calculate probahilities '''
    def __init__(self, wayp_dict, dest_dict, threshold):
        # copy vals
        self.wayp_dict = wayp_dict
        self.dest_dict = dest_dict
        self.threshold = threshold

        # do necessary checks 
        all_point_names = self.waypoint_names + self.destination_names
        assert len(set(all_point_names)) == len(all_point_names)

        for point in self.waypoint_vals + self.destination_vals:
            assert np.array(point).shape == (2,)
        
        # initiate the transition matrices
        n_points = len(wayp_dict) + len(dest_dict)
        self.trans_mat = self.__init_trans_mat(n_points)
        self.trans_mat_normed = self.trans_mat    
        self.trans_mats = self.__init_trans_mat_list(n_points)

    def __init_trans_mat(self, n_points):
        ''' initialize the transition matrix with zeros '''
        return np.zeros((n_points, n_points))

    def __init_trans_mat_list(self, n_points):
        ''' initialize the list of transition matrices, will contain 1 ... n step transition matrices in order to not always need to recalculate it '''

        # estimate of how long paths usually are: 2*sqrt(n_points)
        num_mats = floor(sqrt(len(self.points_names)) * 2)
        trans_mats = [None] * num_mats

        return trans_mats


    @classmethod
    def from_matrices(cls, mat_wayp, mat_dest, threshold):
        ''' Init from matrices '''
        wayp_dict = {}
        dest_dict = {}
        num_waypoints = mat_wayp.shape[0]
        num_destinations = mat_dest.shape[0]
        for i in range(num_waypoints):
            key = 'w{my_i:x}'.format(my_i=i)
            wayp_dict[key] = (mat_wayp[i,0], mat_wayp[i,1])
        for i in range(num_destinations):
            key = 'd{my_i:x}'.format(my_i=i)
            dest_dict[key] = (num_destinations[i,0], num_destinations[i,1])
        return cls(wayp_dict, dest_dict, threshold)  

    def analyse_full_signal(self, path, add_to_trans_mat):
        ''' return a list of waypoints/destinations for a list of location measurements. For each 
        measurement, closest waypoint within threshold range is added. No repetitions 
        of same waypoint allowed. '''
        path_len = len(path)
        # get distance signals
        dist_sig_dict = self.__get_dist_signal_dict(path)
        dist_sigs = np.array(list(dist_sig_dict.values()))

        # get closest waypoint at each moment
        min_dist_ids = np.argmin(dist_sigs, axis=0)
        accessed_points = [None]*path_len

        point_names = list(self.points_dict.keys())
        for i in range(path_len):
            accessed_points[i] = point_names[min_dist_ids[i]]

        # filter based on threshold
        minima = dist_sigs[min_dist_ids, list(range(len(min_dist_ids)))]
        close_enough = minima < self.threshold

        if len(close_enough) < 1:
            return np.array([]) 

        accessed_points_f = np.array(accessed_points)[close_enough]

        # filter to not include series of the same points following each other
        accessed_points_f_scew = np.roll(accessed_points_f, 1)
        accessed_points_f_scew[0] = -1 # in case last point is equal to first
        repetitions = np.not_equal(accessed_points_f, accessed_points_f_scew, dtype = object)
        
        wayp_path = accessed_points_f[repetitions]

        if add_to_trans_mat:
            self.__add_path_to_trans_mat(wayp_path)

        return wayp_path

    def normalize_trans_mat(self):
        ''' normalize the transition matrix '''
        trans_mat_normed = np.zeros_like(self.trans_mat)
        for i in range(len(self.trans_mat)):
            row = self.trans_mat[i]
            row_sum = np.sum(row) 
            if row_sum == 0.:
                warnings.warn("dead points detected", Warning)
                trans_mat_normed[i] = row
            else:
                trans_mat_normed[i] = row/row_sum
        
        self.trans_mat_normed = trans_mat_normed

    def calculate_prob(self, start, end, detour_factor = 0.2):
        ''' using the graph, calculate probability of traveling from start to end node '''
        # make sure that given points are existing
        assert wayp_from in self.points_names
        assert wayp_to in self.points_names
        # get average distance of the most significant paths within the graph
        sign_distances, avg_dist = self.__get_avg_distances()

        # get l1 distance between start and end point
        start_coor = points_dict[start]
        end_coor = points_dict[end]
        l1_travel_dist = np.linalg.norm((start_coor-end_coor), ord=1)

        # get some factors that increase allowed number of steps
        dist_std = np.std(sign_distances) # to incorporat effect of difference in length between nodes

        total_added_factor = (detour_factor + dist_std/avg_dist) * l1_travel_dist 

        #TODO: check whether some lower limit should be set or whether it is unnecesary 

        return None

    def __calculate_prob_n_steps(self, start_node, end_node, min_steps, max_steps, recalculate_mats = True):
        # recalculation can be done first, in case some observations have been added to trans_mat
        if recalculate_mats:
            self.__recalculate_trans_mat_dependencies()

        # make sure steps are integers
        min_steps_r = floor(min_steps)
        max_steps_r = ceil(max_steps)

        # get the probability
        #(I will omit the min_steps for now, it should not be necessary?)

    def visualise_graph():
        return None


    def __recalculate_trans_mat_dependencies():
        ''' recalculate the matrices which depend on the main transition matrix '''
        self.normalize_trans_mat()
        self.__recalculate_trans_mats_list()

    def __recalculate_trans_mats_list(self):
        ''' recalculate the matrices in the transition matrices list '''
        num_matrices = len(self.trans_mats)
        trans_mats = [None] * num_matrices

        last_trans_mat = np.eye(len(self.points_names))
        for i in range(num_matrices):
            last_trans_mat = np.dot(last_trans_mat, self.trans_mat_normed)
            trans_mats[i] = last_trans_mat

        self.trans_mats = trans_mats

    def __get_avg_distances(self):
        ''' calculate the average of distances between all significantly linked nodes '''
        significant_prob = 1/len(self.points_names)
        distance_mat = self.__calculate_node_distances()

        # get distances of paths between nodes that are used significantly much
        significant_dists = np.extract(distance_mat > significant_prob, distance_mat)

        return significant_dists, np.average(significant_dists)

    def __add_path_to_trans_mat(self, wayp_arr):
        ''' add path with multiple waypoint transitions to transition matrix'''
        for i in range(len(wayp_arr)-1):
            self.__add_to_trans_mat(wayp_arr[i], wayp_arr[i+1])

    def __add_to_trans_mat(self, wayp_from, wayp_to):
        ''' add transition between 2 waypoints to transition matrix '''
        assert wayp_from in self.points_names
        assert wayp_to in self.points_names
        i_from = self.points_indices_dict[wayp_from]
        i_to = self.points_indices_dict[wayp_to]

        self.trans_mat[i_from, i_to] += 1


    def __find_closest_waypoint(self, measurement):
        ''' get the closest waypoint to a measurement ''' 
        measurement = np.array(measurement)
        assert measurement.size == 2

        dist_dict = self.__get_dist_signal_dict(measurement)
        dist_arr = np.array(list(dist_dict.values()))

        min = np.argmin(dist_arr)
        waypoint = list(dist_dict.keys())[min]
        distance = dist_arr[min]

        return waypoint, distance[0]

    def measurement_to_waypoint(self, measurement):
        ''' Check closest waypoint that is within threshold, return None if no waypoint '''
        
        w, d = self.__find_closest_waypoint(measurement)

        if d <= self._threshold:
            return w
        else:
            return None

    def __get_dist_signal_dict(self, path):
        ''' for a path, get the distance to all nodes on each time step, later used to find the closest adjacent node '''
        ''' OUT: axis 0: path_num, axis 1: waypoints '''
        if len(path.shape) == 1:
            path = np.expand_dims(path, 0)
        nodes_locations = np.array(list(self.points_dict.values()))
        dist_sig = self.__calculate_node_distances(path, nodes_locations)
        dist_sig_dict = dict(zip(self.points_dict.keys(), dist_sig.T))
        return dist_sig_dict

    def __calculate_node_distances(node_list_1, node_list_2):
        return np.sqrt((node_list_1**2).sum(axis=1)[:, None] - 2 * node_list_1.dot(node_list_2.transpose()) + ((node_list_2**2).sum(axis=1)[None, :]))

    @property  
    def threshold(self):
        return self._threshold
    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    @property 
    def waypoint_names(self):
        return list(self.wayp_dict.keys())

    @property 
    def destination_names(self):
        return list(self.dest_dict.keys())

    @property 
    def waypoint_vals(self):
        return list(self.wayp_dict.values())

    @property 
    def destination_vals(self):
        return list(self.dest_dict.values())

    @property 
    def points_dict(self):
        return {**self.wayp_dict, **self.dest_dict}

    @property
    def points_names(self):
        return list(self.points_dict.keys())

    @property 
    def points_indices_dict(self):
        return dict(zip(self.points_names, range(len(self.points_names))))

# Test function for module  
def _test():
    ''' test graph class ''' 
    wayp_dict = {'w1': [1.,1.], 'w2': [2., 2.]}
    dest_dict = {'d1': [0.,0.], 'd2': [3., 3.]}
    threshold = .5
    g = Graph(wayp_dict, dest_dict, threshold)
    path = np.array([[0., 0.], [1., 1.], [2.1,2.], [2.1,2.]])
    a = g.analyse_full_signal(path)

    return None

if __name__ == '__main__':
    _test()