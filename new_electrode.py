import yaml, os, MEAutility 

import numpy as np 

ne = 16
space = 30


def compute_positions(ne,space):
    start = -space * (ne-1)/2
    steps = [start + space*k for k in range (ne)]
    pos = [[0]*3 for i in range (ne*ne)]
    e = 0
    for i in range(ne) :
        for j in range(ne) :
            pos[e][1] = steps[i]
            pos[e][2] = steps[j]
            e += 1
    return pos


pos = compute_positions(ne,space)

user_info = {'dim': ne,
             'electrode_name': 'Fake_probe',
             'description': "Probe similar to the one for real data",
             'pitch': space,
             'shape': 'square',
             'size': 5,
             'sortlist': None,
             'stagger': [0, -12, 30, -22],
             'type': 'mea',
             'position': pos}


with open('user.yaml', 'w') as f:
    yaml.dump(user_info, f)

yaml_files = [f for f in os.listdir('.') if f.endswith('.yaml')]

MEAutility.add_mea('user.yaml')



