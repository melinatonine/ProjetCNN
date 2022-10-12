import MEArec as mr
import MEAutility as mu
import yaml
import os
import numpy as np
from pprint import pprint
from pathlib import Path
import matplotlib.pylab as plt
from pprint import pprint

cell_folder = mr.get_default_cell_models_folder()
template_params = mr.get_default_templates_params()

drifting = False
erase = False

# let's generate 10 templates per cell models (total 130 templates)
template_params["n"] = 50

template_params["drifting"] = True
template_params["drift_steps"] = 30
# this ensures that all cells drift on the same z trajectory, with a small xy variation
template_params["drift_xlim"] = [-10, 10]
template_params["drift_ylim"] = [-10, 10]
template_params["drift_zlim"] = [20, 80]
template_params["max_drift"] = 200


probes = ['Neuronexus-32']
n_jobs = 6

for probe in probes:
    print(f"Generating drifting templates for {probe}")
    template_params["probe"] = probe
    filename = f"templates_drift_{probe}.h5"
    
    if not Path(filename).is_file():
    
        tempgen = mr.gen_templates(cell_models_folder=cell_folder, params=template_params, 
                                   n_jobs=n_jobs, verbose=True, recompile=True)

        mr.save_template_generator(tempgen, filename=filename)
    else:
        print(f"{filename} already generated")

tempgens = {}

for probe in probes:
    filename = f"templates_drift_{probe}.h5"
    
    tempgen = mr.load_templates(filename)
    tempgens[probe] = tempgen


# set duration and number of units

# 10 min
recordings_params = mr.get_default_recordings_params()
recordings_params["spiketrains"]["duration"] = 60*5

# 100 Excitatory, 20 inhibitory (the main difference is morphology and avg firing rates)
recordings_params["spiketrains"]["n_exc"] = 20
recordings_params["spiketrains"]["n_inh"] = 5
recordings_params["templates"]["min_amp"] = 30
recordings_params["templates"]["min_dist"] = 20 # um 
recordings_params["recordings"]["filter"] = False
recordings_params["recordings"]["noise_level"] = 0
recordings_params["recordings"]["noise_mode"] = "distance-correlated"
recordings_params['recordings']['chunk_duration'] = 1.

# (optional) set seeds for reproducibility 
# (e.g. if you want to maintain underlying activity, but change e.g. noise level)
recordings_params['seeds']['spiketrains'] = 42
recordings_params['seeds']['templates'] = 42
recordings_params['seeds']['convolution'] = 42
recordings_params['seeds']['noise'] = 42

# no drift for now (first location is used)
if not drifting:
    recordings_params["recordings"]["drifting"] = False
else:
    recordings_params["recordings"]["drifting"] = True
    recordings_params["recordings"]["drift_mode"] = "slow"
    recordings_params["recordings"]["slow_drift_velocity"] = 5 # um/min

n_jobs = -1

for probe, tempgen in tempgens.items():
    if drifting:    
        filename = f"recordings_{probe}_noisy_drifting.h5"
    else:
        filename = f"recordings_{probe}_nonoise_static.h5"

    if erase or not os.path.exists(filename):
        recgen = mr.gen_recordings(params=recordings_params, tempgen=tempgen, 
                                   n_jobs=n_jobs, verbose=True)
        mr.save_recording_generator(recgen, filename=filename)