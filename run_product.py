import numpy as np
from itertools import product

from os import makedirs

import subprocess

params = {
        '--model': ['munet'],
        '--epochs': [10000],
        '--batch_size': [1,2,4],
        '--workers': [10],
        '--lr': np.logspace(-6, -2, 5),
        '--brightness': [0.1],
        '--contrast': [0.1],
        '--lmbda': [0.5],
        '--weight_decay': np.logspace(-4, -2, 3)
        }

ll = []

for key, vals in params.items():
    l = []
    
    for val in vals:  
        l.append((key, val))
    
    ll.append(l)
    
prod = product(*ll)

k = 0

makedirs('run_files', exist_ok=True)

for tuples in prod:
    path = 'run_files/run'+str(k)+'.sh'
    f = open(path, "w")
    s = 'pipenv run python main.py '
    for (a, b) in tuples:
        s = s + ' ' + a + ' ' + str(b)
        
    s = '#!/usr/bin/bash\n\
module load gcc/8.2.0 python_gpu/3.8.5\n\
export HTTP_PROXY=http://proxy.ethz.ch:3128\n\
export HTTPs_PROXY=http://proxy.ethz.ch:3128\n\
nvidia-smi\n\
pipenv install\n' + s + ' --img_dim 400 --h_flip --v_flip --rotate --save_dir save/ --data /cluster/scratch --train_data cil --val_data cil  --val_period 5'
    f.write(s)
    f.close()
    subprocess.run(['bsub -n 10 -W 3:59 -R "rusage[mem=2000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=9000]" < ' + path])
    k= k + 1
