import json, os, sys
import logging
from tracks import CustomTrack

json_logger = logging.getLogger('json')
json_logger.setLevel(logging.INFO)

PERSON = {
    "id": 0,
    "gender" : "",
    "bag" : False,
    "hat" : False,
    "upper_color" : "",
    "lower_color" : "",
    "roi1_passages" : 0,
    "roi1_persistence_time" : 0,
    "roi2_passages" : 0,
    "roi2_persistence_time" : 0
}

PEOPLE = {
    "people" : []
}

class FileJson():

    def __init__(self, path):
        self._path = os.path.join(os.path.dirname(__file__), path)
    
    def read_roi(self):
        try:
            with open(self._path, "r") as f:
                file = json.load(f)
                roi1 = file["roi1"]
                roi2 = file["roi2"]
                json_logger.info(f'File {os.path.basename(self._path)} loaded')
        except:
            json_logger.error(f'File {os.path.basename(self._path)} not found')
            sys.exit(1)
        
        return roi1, roi2
    
    def write_par(self, par_dict : dict):
        data = PEOPLE.copy()
        par_dict = dict(sorted(par_dict.items()))

        v : CustomTrack
        for _, v in par_dict.items():
            person = PERSON.copy()
            
            person["id"] = v.track_id
            person["gender"] = v._gender
            person["bag"] = v._bag
            person["hat"] = v._hat
            person["upper_color"] = v._upper
            person["lower_color"] = v._lower
            person["roi1_passages"] = v._roi1_transit
            person["roi1_persistence_time"] = v._roi1_time
            person["roi2_passages"] = v._roi2_transit
            person["roi2_persistence_time"] = v._roi2_time

            data["people"].append(person)
        
        try:
            with open(self._path, 'w') as f:
                json.dump(data, f, indent=4)
                json_logger.info(f'File {os.path.basename(self._path)} written')
        except:
            json_logger.error(f'File {os.path.basename(self._path)} not written')
            sys.exit(1)
   












