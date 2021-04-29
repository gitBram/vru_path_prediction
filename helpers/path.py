''' Path class which contains the path of one pedestrian. '''
import numpy as np

class Path:
    def __init__(self, measurement_arr, graph):
        ''' entity to keep track during runtime of all measurements for one path and the assiociated waypoints '''
        # check shape of input  
        assert measurement_arr.size > 0

        # store the measurements and graph within the class
        self.meas_arr = measurement_arr
        self.graph = graph
        self.wayp_arr = None 

    def make_waypoint_arr(self, add_to_trans_mat):
        ''' use a graph class with defined waypoints/destinations to convert the internal measurements to waypoints '''
        self.wayp_arr = self.graph.analyse_full_signal(self.meas_arr, add_to_trans_mat)

    def add_measurement(self, measurement, update_waypoints = True):
        ''' add a measurement to the meas_arr, for online use of class '''
        self.meas_arr = np.append(self.meas_arr, measurement)   

        if update_waypoints:
            self.__add_waypoint_from_measurement(measurement)

    def __add_waypoint_from_measurement(self, measurement):
        w = self.graph.measurement_to_waypoint(measurement)
        self.__add_waypoint(w)

    def __add_waypoint(self, waypoint):
        ''' Add waypoint if it's not the same as the last one '''
        try:
            # will fail if length of wayp_path == 0
            if self.wayp_path[-1] != waypoint:
                self.wayp_arr = np.append(self.wayp_arr, waypoint)
        except:
            self.wayp_arr = np.append(self.wayp_arr, waypoint)

    def __str__(self):
        return str(self.wayp_arr)

    def __repr__(self):
        return str(self.wayp_arr)

    # @property  
    # def wayp_path(self):
    #     return self.wayp_path

# Test function for module  
def _test():
      
    from graph import Graph
    wayp_dict = {'w1': [1.,1.], 'w2': [2., 2.]}
    dest_dict = {'d1': [0.,0.], 'd2': [3., 3.]}
    threshold = .5
    g = Graph(wayp_dict, dest_dict, threshold)

    path = np.array([[0., 0.], [1., 1.], [2.1,2.], [2.1,2.]])

    p = Path(path, g)  
    p.make_waypoint_arr(True)
    print(p.wayp_arr)
    p.add_measurement([3., 3.], True)
    print(p.wayp_arr)
    print(g.trans_mat)
    # assert np.all(p.wayp_path == np.array([1, 2, 3]))


if __name__ == '__main__':
    _test()


