from deep_sort_realtime.deep_sort.track import Track

class CustomTrack(Track):

    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None, original_ltwh=None, det_class=None, det_conf=None, instance_mask=None, others=None):
        super().__init__(mean, covariance, track_id, n_init, max_age, feature, original_ltwh, det_class, det_conf, instance_mask, others)

        self._upper = None
        self._lower = None
        self._gender = None
        self._bag = None
        self._hat = None

        self._roi1_time = 0
        self._roi2_time = 0

        self._roi1_transit = 0
        self._roi2_transit = 0

        self._roi1_inside = False
        self._roi2_inside = False
