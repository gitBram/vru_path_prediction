import numpy as np

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

    def __init_trans_mat(self, n_points):
        return np.zeros((n_points, n_points))

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
    
    def calculate_prob(start, end, n_steps):
        ''' '''
        bullshit here
        return None

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
        ''' axis 0: path_num, axis 1: waypoints '''
        if len(path.shape) == 1:
            path = np.expand_dims(path, 0)
        waypoints = np.array(list(self.points_dict.values()))
        dist_sig = np.sqrt((path**2).sum(axis=1)[:, None] - 2 * path.dot(waypoints.transpose()) + ((waypoints**2).sum(axis=1)[None, :]))
        dist_sig_dict = dict(zip(self.points_dict.keys(), dist_sig.T))
        return dist_sig_dict

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