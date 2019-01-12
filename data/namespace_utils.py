import json
from bunch import Bunch

def save_namespace(FLAGS, out_path):
    FLAGS_dict = vars(FLAGS)
    with open(out_path, 'w') as fp:
        #json.dump(FLAGS_dict, fp)
        json.dump(FLAGS_dict, fp, indent=4, sort_keys=True)
        
def load_namespace(in_path):
    with open(in_path, 'r') as fp:
        FLAGS_dict = json.load(fp)
        
        if FLAGS_dict.get("add_position_timing_signal", False) == True:
            FLAGS_dict["pos"] = None
        for key in FLAGS_dict:
            print(key, FLAGS_dict[key])
    return Bunch(FLAGS_dict)