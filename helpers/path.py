''' Path class which contains the path of one pedestrian. '''
import numpy as np

class Path:
    def __init__(self, waypoint_arr = np.array([])):
        ''' define from series of observed waypoints, filter out double observations '''

        # check shape of input
        assert len(waypoint_arr.shape) <= 1
        

        # will fail if waypoint_arr is empty
        try:
            # filter to not include series of the same points following each other
            accessed_points_f_scew = np.roll(waypoint_arr, 1)
            accessed_points_f_scew[0] = -1 # in case last point is equal to first
            # print(accessed_points_f_scew)
            repetitions = np.not_equal(waypoint_arr, accessed_points_f_scew)
            
            # print(repetitions)
            wayp_path = waypoint_arr[repetitions]
            # print(wayp_path)

            self.wayp_path = wayp_path
        except:
            self.wayp_path = np.array([])

    def add_waypoint(self, waypoint):
        ''' Add waypoint if it's not the same as the last one '''
        try:
            # will fail if length of wayp_path == 0
            if self.wayp_path[-1] != waypoint:
                self.wayp_path = np.append(self.wayp_path, waypoint)
        except:
            self.wayp_path = np.append(self.wayp_path, waypoint)

    def __str__(self):
        return str(self.wayp_path)

    def __repr__(self):
        return str(self.wayp_path)

    # @property  
    # def wayp_path(self):
    #     return self.wayp_path

# Test function for module  
def _test():
    p = Path(np.array([1, 2, 3]))
    assert np.all(p.wayp_path == np.array([1, 2, 3]))

if __name__ == '__main__':
    _test()


