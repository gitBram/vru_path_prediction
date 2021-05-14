import numpy as np
import warnings
from math import floor, ceil, sqrt
import matplotlib.pyplot as plt 
import networkx as nx

# TODO: Add observers in order to only do the necessary recalculations https://stackoverflow.com/questions/6190468/how-to-trigger-function-on-value-change

class Graph():
    ''' Graph class to fill transition matrix and calculate probahilities '''
    def __init__(self, wayp_dict, dest_dict, dist_threshold, std_graph_prune_threshold):
        # copy vals
        self.wayp_dict = wayp_dict
        self.dest_dict = dest_dict
        self.dist_threshold = dist_threshold
        self.std_graph_prune_threshold = std_graph_prune_threshold

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

        # placeholder for the standard graph
        self.std_graph = None

    def __init_trans_mat(self, n_points):
        ''' initialize the transition matrix with zeros '''
        return np.zeros((n_points, n_points))

    def __init_trans_mat_list(self, n_points):
        ''' initialize the list of transition matrices, will contain 1 ... n step transition matrices in order to not always need to recalculate it '''
        # in current implementation --> just empty dict which is filled 
        return dict()


    @classmethod
    def from_matrices(cls, mat_wayp, mat_dest, dist_threshold, std_graph_prune_threshold):
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
            dest_dict[key] = (mat_dest[i,0], mat_dest[i,1])
        return cls(wayp_dict, dest_dict, dist_threshold, std_graph_prune_threshold)  


    def analyse_multiple_full_signals(self, signal_list, add_to_trams_mat):
        return_lists = []

        for signal in signal_list:
            return_lists.append(self.analyse_full_signal(signal, add_to_trams_mat))

        return return_lists

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
        accessed_points = [None] * path_len

        point_names = list(self.points_dict.keys())
        for i in range(path_len):
            accessed_points[i] = point_names[min_dist_ids[i]]

        # filter based on threshold
        minima = dist_sigs[min_dist_ids, list(range(len(min_dist_ids)))]
        close_enough = minima < self.dist_threshold

        if len(close_enough) < 1:
            return np.array([]) 

        accessed_points_f = np.array(accessed_points)[close_enough]

        try:
            # filter to not include series of the same points following each other
            accessed_points_f_scew = np.roll(accessed_points_f, 1)
            accessed_points_f_scew[0] = -1 # in case last point is equal to first
            repetitions = np.not_equal(accessed_points_f, accessed_points_f_scew, dtype = object)
            
            wayp_names = accessed_points_f[repetitions]
            wayp_locs = self.__point_names_to_locs(wayp_names)

            if add_to_trans_mat:
                self.__add_path_to_trans_mat(wayp_names)

            return wayp_names, wayp_locs
        except:
            warnings.warn("Unused path in transition matrix due to not reaching two waypoints", Warning)
            return [], []

        

    def __point_names_to_locs(self, point_names):
        locs = []
        for point_name in point_names:
            locs.append(self.points_dict[point_name])
        return locs

    def __normalize_trans_mat(self):
        ''' normalize the transition matrix '''
        trans_mat_normed = np.zeros_like(self.trans_mat)
        for i in range(len(self.trans_mat)):
            row = self.trans_mat[i]
            row_sum = np.sum(row) 
            if row_sum == 0.:
                warnings.warn("dead points detected: noone leaving from point %s"%(self.indices_points_dict[i]), Warning)
                trans_mat_normed[i] = row
            else:
                trans_mat_normed[i] = row/row_sum
        
        self.trans_mat_normed = trans_mat_normed

    def calculate_destination_probs(self, path):
        start_node = path[0]
        end_node = path[-1]
        # prob from start to end of path
        path_prob = self.calculate_path_prob(path)
        # prob from end to every destination
        end_to_dest_prob = dict()
        for dest in self.destination_names:
            end_to_dest_prob[dest] = self.calculate_prob(end_node, dest)
        # prob from start of path to every destination
        start_to_dest_prob = dict()
        for dest in self.destination_names:
            start_to_dest_prob[dest] = self.calculate_prob(start_node, dest)
        # full prob
        full_prob = dict()
        for dest in self.destination_names:
            full_prob[dest] = path_prob * end_to_dest_prob[dest] / start_to_dest_prob[dest]
        
        print('Path prob:')
        print(path_prob)
        print('Path end to Dest prob:')
        print(end_to_dest_prob)
        print('Path start to Dest prob:')
        print(start_to_dest_prob)
        print('Full calculated probs:')
        print(full_prob)
        return full_prob

    def calculate_path_prob(self, path):
        # basic content check
        for point in path:
            assert point in self.points_names
        
        total_prob = 1.
        for i in range(1, len(path)):
            start_node = path[i-1]
            end_node = path[i]
            total_prob *= self.__calculate_prob_n_steps(start_node, end_node, 1, 1)
        return total_prob

    def calculate_prob(self, start_node, end_node, detour_factor = 0.2, dist_calc_method = 'matrices'):
        ''' using the graph, calculate probability of traveling from start to end node '''
        # make sure that given points are existing
        assert start_node in self.points_names
        assert end_node in self.points_names

        min_steps, max_steps = self.__min_max_step_num(start_node, end_node, method=dist_calc_method)
        #TODO: check whether some lower limit should be set or whether it is unnecesary 

        return self.__calculate_prob_n_steps(start_node, end_node, min_steps, max_steps)

    def __min_max_step_num(self, start_node, end_node, detour_factor = .2, method = 'matrices'):
        ''' calculate the minimum and maximum steps between 2 points '''
        ''' method can be either graph based or distance based '''
        # get indices
        start_node_id = self.points_indices_dict[start_node]
        end_node_id = self.points_indices_dict[end_node]

        if method == 'distance':
            # get average distance of the most significant paths within the graph
            sign_distances, avg_dist = self.__get_avg_distances()

            # get l1/l2 distance between start and end point
            start_coor = np.array(self.points_locations[start_node_id])
            end_coor = np.array(self.points_locations[end_node_id])
            l1_travel_dist = np.linalg.norm((start_coor-end_coor), ord=1)
            l2_travel_dist = np.linalg.norm((start_coor-end_coor))

            # get some factors that increase allowed number of steps
            dist_std = np.std(sign_distances) # to incorporat effect of difference in length between nodes

            total_added_factor = (detour_factor + dist_std/avg_dist) * l1_travel_dist 

            # finally decide minimum and maximum number of steps for certain trajectory
            min_steps = floor(max(1, l2_travel_dist/avg_dist))
            max_steps = ceil(l1_travel_dist + total_added_factor)
        elif method == 'graph': # method is graph based
            try:
                graph = self.std_graph
                shortest_path = nx.shortest_path(graph, start_node_id, end_node_id) 
                min_steps = len(shortest_path) - 1 
                max_steps = ceil((1+detour_factor)*min_steps)
            except:
                warnings.warn("No viable path found between %s and %s"%(start_node, end_node), Warning)
                min_steps = 0
                max_steps = 0
        elif method == 'matrices':
            # Let's say that we expect paths not to be longer than 10 steps
            max_n_steps = 10

            # now let's check after how many steps end_point can be reached from start_point
            for i in range(1,max_n_steps + 1):
                prob = self.__calculate_prob_n_steps(start_node, end_node, i, i)
                # prob = self.trans_mats_dict[i][start_node_id, end_node_id]

                min_steps = 0
                max_steps = 0

                if prob > 0.0:
                    min_steps = i
                    max_steps = ceil((1+detour_factor)*i)

        return min_steps, max_steps

    def __calculate_prob_n_steps(self, start_node, end_node, min_steps, max_steps, recalculate_mats = True):
        # in case there was no shortest route found, no chance of getting there so zero probability returned
        if min_steps == 0. and max_steps == 0.:
            return 0.
        
        # recalculation can be done first, in case some observations have been added to trans_mat
        if recalculate_mats:
            self.recalculate_trans_mat_dependencies()

        # make sure steps are integers
        min_steps_r = floor(min_steps)
        max_steps_r = ceil(max_steps)

        # get the id's of the start and end node in the matrix
        start_id = self.points_indices_dict[start_node]
        end_id = self.points_indices_dict[end_node]

        # for convenience...
        n_stored_ms = self._num_stored_trans_mats

        # get the probability
        current_prob = 0.
        for step in range(min_steps_r, max_steps_r + 1):
            if not step in self.stored_mats_keys:
                # we do not have precalculated matrix --> calculate from last matrix in trans_mats
                self.trans_mats_dict[step] = np.dot(self.trans_mats_dict[n_stored_ms], 
                    self.__np_dot_square(self.trans_mat_normed, step - n_stored_ms))
                
            # calculate the prob
            prob = self.trans_mats_dict[step][start_id, end_id]
            # multiply with the prob
            current_prob += prob
        return current_prob

    def create_graph(self, threshold = 0.):
        ''' create a NetworkX based graph of the current status, allowing use of in-built analysis functions '''
        # normalized transition matrix is needed here to prune the graph based on threshold
        trans_mat = self.trans_mat_normed
        
        # initialize graph
        G = nx.DiGraph()
        inv_wayp = {y:x for x,y in self.destinations_indices_dict.items()}
        inv_dest = {y:x for x,y in self.waypoints_indices_dict.items()}
        G.add_nodes_from(inv_wayp)
        G.add_nodes_from(inv_dest)
        # create the nodes with weight from 
        for start_node in self.points_names:
            for end_node in self.points_names:
                if start_node == end_node:
                    continue
                start_node_id = self.points_indices_dict[start_node]
                end_node_id = self.points_indices_dict[end_node]
                weight = trans_mat[start_node_id, end_node_id]

                if weight > threshold:
                    G.add_edge(start_node_id, end_node_id)
                    # print('start: %s (%d) end: %s (%d) value: %d' % (start_node, start_node_id, end_node, end_node_id, weight))
   
        return G

    def visualize_graph(self, graph, save_loc, g_type = 'relative', scene_loader = None):
        ''' visualize graph (nodes and edges with its weights, absolute or relative ) '''

        # remove the possibly currently existing plot
        plt.clf()

        # if available, plot the image of the scene in the background
        if not scene_loader is None:
            scene_loader.plot_dests_on_img(save_loc)

        #check inputs
        assert save_loc is not None
        assert g_type == 'absolute' or g_type  == 'relative'

        # get relative or absolute weight numbers
        trans_mat = self.trans_mat_normed if g_type == 'relative' else self.trans_mat

        # get number of nodes
        num_nodes = self.num_nodes

        # get node locations for drawing on x,y coordinate
        node_locations = self.points_locations
      
        # we have to collect weights in a separate loop as networkx does not add edges in order
        weights = []
        for edge in graph.edges():
            weight = trans_mat[edge[0], edge[1]]
            weights.append(weight)
        print(graph.edges)
        # set node color depending on destination or waypoint
        d_c = 'red'
        w_c = 'green'
        node_colors = []
        for n in graph.nodes():
            if n in self.waypoints_indices_dict.values():
                node_colors.append(w_c)
            else:
                node_colors.append(d_c)

        options = {
            "node_color": node_colors,
            "edge_color": weights,
            "width": 1,
            "edge_cmap": plt.cm.autumn,
            "with_labels": False,
            "connectionstyle":'arc3, rad = 0.1'
        }
        graph_fig = nx.draw(graph, node_locations, **options)
        inv_points_ids = {y:x for x,y in self.points_indices_dict.items()}
        higher_pos = [[l[0], l[1]+.1] for l in node_locations]
        nx.draw_networkx_labels(graph,higher_pos,inv_points_ids,font_size=12,font_color='black')
        plt.savefig(save_loc)

    def __np_dot_square(self, matrix, power):
        ''' dot product of matrix with itself power amount of times '''
        if power == 1:
            return matrix
        else:
            mats = [matrix for i in range(power)]
            return np.linalg.multi_dot(mats)

    def recalculate_trans_mat_dependencies(self):
        ''' recalculate the matrices which depend on the main transition matrix '''
        self.__normalize_trans_mat()
        self.__recalculate_trans_mats_list()
        self.__recalculate_std_graph()

    def __recalculate_std_graph(self):
        self.std_graph = self.create_graph(self.std_graph_prune_threshold)


    def __recalculate_trans_mats_list(self):
        ''' recalculate the matrices in the transition matrices dictionary '''
        num_matrices = self._num_stored_trans_mats
        trans_mats = dict()

        last_trans_mat = np.eye(len(self.points_names))
        for i in range(1, num_matrices + 1):
            last_trans_mat = np.dot(last_trans_mat, self.trans_mat_normed)
            trans_mats[i] = last_trans_mat

        self.trans_mats_dict = trans_mats

    def __get_avg_distances(self):
        ''' calculate the average of distances between all significantly linked nodes '''
        significant_prob = 1/len(self.points_names)
        locs = np.array(self.points_locations)
        distance_mat = self.__calculate_node_distances(locs, locs)

        # get distances of paths between nodes that are used significantly much
        significant_dists = np.extract(self.trans_mat_normed > significant_prob, distance_mat)

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
        ''' calculate distances '''
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
    def indices_points_dict(self):
        return dict(zip(range(len(self.points_names)), self.points_names))
    
    @property 
    def waypoints_indices_dict(self):
        ''' waypoints get the 0-n_waypoints indices ''' 
        return dict(zip(self.waypoint_names, list(range(self.num_waypoints))))

    @property 
    def destinations_indices_dict(self):
        ''' destinations get (n_waypoints+1) - (n_waypoints+n_destinations) indices'''
        return dict(zip(self.destination_names, 
        [x + self.num_waypoints for x in list(range(self.num_destinations))]))

    @property
    def num_nodes(self):
        return len(self.points_names)
    
    @property 
    def num_destinations(self):
        return len(self.destination_names)
    
    @property 
    def num_waypoints(self):
        return len(self.waypoint_names)

    @property 
    def stored_mats_keys(self):
        return list(self.trans_mats_dict.keys())

# Test function for module  
def _test():
    ''' test graph class ''' 
    wayp_dict = {'w1': [1.,1.], 'w2': [2., 1.], 'w3': [3., 2.], 'w4': [3., 0.]}
    dest_dict = {'d1': [0.,1.], 'd2': [4., 2.], 'd3': [4., 0.]}
    threshold = .5
    prune_threshold = .05
    g = Graph(wayp_dict, dest_dict, threshold, prune_threshold)
    path1 = np.array([[0., 1.], [1., 1.], [2.,1.], [3.,2.], [4.,2.]])
    path2 = np.array([[0., 1.], [1., 1.], [2.,1.], [3.,0.], [4.,0.]])
    path3 = np.array([[0., 1.], [1., 1.], [2.,1.], [3.,0.], [4.,0.]])
    path4 = np.array([[4.,0.], [3.,0.], [2.,1.], [1., 1.], [0., 1.]])
    a,l = g.analyse_full_signal(path1, add_to_trans_mat = True)
    a,l = g.analyse_full_signal(path2, add_to_trans_mat = True)
    a,l = g.analyse_full_signal(path3, add_to_trans_mat = True)
    a,l = g.analyse_full_signal(path4, add_to_trans_mat = True)
    print("Locations:")
    print(a)
    print(l)
    g.recalculate_trans_mat_dependencies()
    print(g.waypoints_indices_dict)
    print(g.destinations_indices_dict)
    print(g.trans_mat)
    print(g.trans_mat_normed)
    my_graph = g.create_graph(.05)
    g.visualize_graph(my_graph, './data/images/graph.png', g_type='relative')


    ''' Test for calculate_path_prob '''
    assert g.calculate_path_prob(['d1', 'w1', 'w2', 'w3']) == 1*.75*.25


    return None

if __name__ == '__main__':
    _test()