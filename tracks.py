from deep_sort_realtime.deep_sort.track import Track

COLORS = {0: 'black', 1: 'blue', 2: 'brown', 3: 'gray', 4: 'green', 5: 'orange', 6: 'pink', 7: 'purple', 8: 'red', 9: 'white', 10: 'yellow'}
GENDER = {0 : 'male', 1 : 'female'}
BAG = {0 : False, 1 : True}
HAT = {0 : False, 1 : True}

class CustomTrack(Track):

    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None, original_ltwh=None, det_class=None, det_conf=None, instance_mask=None, others=None):
        super().__init__(mean, covariance, track_id, n_init, max_age, feature, original_ltwh, det_class, det_conf, instance_mask, others)

        self.upper = None
        self.lower = None
        self.gender = None
        self.bag = None
        self.hat = None

        self.roi1_time = 0
        self.roi2_time = 0

        self.roi1_transit = 0
        self.roi2_transit = 0

        self.roi1_inside = False
        self.roi2_inside = False

        self._measurements = dict()
        self._number_measurements = 0
        self._is_par_confirmed = False
        self._N = 15
    
    def add_par_measurement(self, task, pred):
        try:
            self._measurements[task].append(pred)
        except KeyError:
            self._measurements[task] = [pred]
    
    def check_limit_par_measurements(self):
        self._number_measurements += 1
        
        if self._number_measurements >= self._N:
            self.find_max()
            
    def find_max(self):
        pred_dict = dict()
        for k, v in self._measurements.items():
            pred_dict[k] = max(set(v), key = v.count)
        
        self._measurements = self._number_measurements = None

        self.upper = COLORS[pred_dict[0]]
        self.lower = COLORS[pred_dict[1]]
        self.gender = GENDER[pred_dict[2]]
        self.bag = BAG[pred_dict[3]]
        self.hat = HAT[pred_dict[4]]

        self._is_par_confirmed = True
    
    def is_measurements_empty(self):
        return len(self._measurements) == 0



        
