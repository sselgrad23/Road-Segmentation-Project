





#!/usr/bin/bash

module load gcc/8.2.0 python_gpu/3.8.5
export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPs_PROXY=http://proxy.ethz.ch:3128

nvidia-smi

# pipenv install


pipenv run  python main.py --model v2GAN --data /cluster/scratch/zdavid/cil_data_root  --epochs 10000 --batch_size 8 --workers 10 --lr 0.0005 \
     --save_dir save/ --img_dim 400 --h_flip --v_flip --brightness 0.1 --contrast 0.1 --rotate --distort \
    --lmbda 0.5 --train_data cil --val_data cil  --val_period 5  --beta1 0.5 --beta2 0.999
# --rotate
# bsub -n 10 -W 3:59 -R "rusage[mem=2000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=18000]" < run_euler.sh
