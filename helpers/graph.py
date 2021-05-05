import numpy as np
import warnings
from math import floor, ceil, sqrt
import matplotlib.pyplot as plt 
import networkx as nx

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
        self._num_stored_trans_mats = floor(sqrt(len(self.points_names)) * 2) # estimate of how long paths usually are: 2*sqrt(n_points)
        self.trans_mats_dict = self.__init_trans_mat_list(n_points)

    def __init_trans_mat(self, n_points):
        ''' initialize the transition matrix with zeros '''
        return np.zeros((n_points, n_points))

    def __init_trans_mat_list(self, n_points):
        ''' initialize the list of transition matrices, will contain 1 ... n step transition matrices in order to not always need to recalculate it '''
        # in current implementation --> just empty dict which is filled 
        return dict()


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

    def calculate_destination_probs(self, path):
        start_node = path[0]
        end_node = path[-1]
        # prob from start to end of path
        path_prob = self.calculate_prob(start_node, end_node)
        # prob from start to every destination
        partial_dest_prob = dict()
        for dest in self.destination_names:
            partial_dest_prob[dest] = self.calculate_prob(end_node, dest)
        # prob from end of path to every destination
        full_dest_prob = dict()
        for dest in self.destination_names:
            full_dest_prob[dest] = self.calculate_prob(start_node, dest)
        # full prob
        full_prob = dict()
        for dest in self.destination_names:
            full_prob[dest] = path_prob * partial_dest_prob[dest] / full_dest_prob[dest]
        
        return full_prob

    def calculate_prob(self, start_node, end_node, detour_factor = 0.2):
        ''' using the graph, calculate probability of traveling from start to end node '''
        # make sure that given points are existing
        assert start_node in self.points_names
        assert end_node in self.points_names
        # get average distance of the most significant paths within the graph
        sign_distances, avg_dist = self.__get_avg_distances()

        # get l1 distance between start and end point
        start_coor = points_dict[start_node]
        end_coor = points_dict[end_node]
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

        # get the id's of the start and end node in the matrix
        start_id = self.points_indices_dict[start_node]
        end_id = self.points_indices_dict[end_node]

        # for convenience...
        n_stored_ms = self._num_stored_trans_mats

        # get the probability
        current_prob = 1.
        for step in range(min_steps_r, max_steps_r + 1):
            if not step in self.stored_mats_keys:
                # we do not have precalculated matrix --> calculate from last matrix in trans_mats
                self.trans_mats_dict[step] = np.dot(self.trans_mats_dict[n_stored_ms], 
                    self.__np_dot_square(self.trans_mat_normed, step - n_stored_ms))
                
            # calculate the prob
            prob = self.trans_mats_dict[step][start_id, end_id]
            # multiply with the prob
            current_prob *= prob

    def visualize_graph(self, save_loc, threshold = 0.):
        num_nodes = self.num_nodes
        trans_mat_normed = self.trans_mat_normed
        node_locations = self.points_locations

        # initialize graph
        G = nx.DiGraph()

        # create the nodes with weight from 
        weights = []

        for start_node in range(num_nodes):
            for end_node in range(num_nodes):
                if start_node == end_node:
                    continue
                weight = trans_mat_normed[start_node, end_node]

                if weight > threshold:
                    G.add_edge(start_node, end_node)
                    weights.append(weight)

        options = {
            "node_color": "green",
            "edge_color": weights,
            "width": 4,
            "edge_cmap": plt.cm.Blues,
            "with_labels": True,
            "connectionstyle":'arc3, rad = 0.1'
        }
        graph_fig = nx.draw(G, node_locations, **options)

        plt.savefig(save_loc)

    def __np_dot_square(matrix, power):
        ''' dot product of matrix with itself power amount of times '''
        mats = [matrix for i in range(power)]

        return np.linalg.multi_dot(mats)

    def __recalculate_trans_mat_dependencies():
        ''' recalculate the matrices which depend on the main transition matrix '''
        self.normalize_trans_mat()
        self.__recalculate_trans_mats_list()

    def __recalculate_trans_mats_list(self):
        ''' recalculate the matrices in the transition matrices dictionary '''
        num_matrices = self._num_stored_trans_mats
        trans_mats = dict()

        last_trans_mat = np.eye(len(self.points_names))
        for i in range(1, num_matrices + 1):
            last_trans_mat = np.dot(last_trans_mat, self.trans_mat_normed)
            trans_mats[i] = last_trans_mat

        self.trans_mats_dict = trans_mats_dict

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

    def __calculate_node_distances(self, node_list_1, node_list_2):
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
    def points_locations(self):
        return list(self.points_dict.values())

    @property 
    def points_indices_dict(self):
        return dict(zip(self.points_names, range(len(self.points_names))))

    @property
    def num_nodes(self):
        return len(self.points_names)
    
    @property 
    def stored_mats_keys(self):
        return list(self.trans_mats_dict.keys())

# Test function for module  
def _test():
    ''' test graph class ''' 
    wayp_dict = {'w1': [1.,1.], 'w2': [2., 2.]}
    dest_dict = {'d1': [0.,0.], 'd2': [2., 3.]}
    threshold = .5
    g = Graph(wayp_dict, dest_dict, threshold)
    path = np.array([[0., 0.], [1., 1.], [2.1,2.], [2.1,2.]])
    a = g.analyse_full_signal(path, add_to_trans_mat = True)
    g.visualize_graph('./data/images/graph.png')
    return None

if __name__ == '__main__':
    _test()