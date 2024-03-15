#!/usr/bin/bash



#For loop stufff
model="v2GAN"
#lr=('0.00001' '0.00005' '0.0001' '0.0005' '0.001')
learning_rate=(0.00005 0.0001 0.0005 0.001)
lmbda=(0.5 0.9 1.0)
loss=(dice bce)
#weight_decay=(0.00001)
#version=(1)
#pipenv install

for lr in "${learning_rate[@]}"
do
  for lambda in "${lmbda[@]}"
  do
    for l in "${loss[@]}"
      do
        bsub -n 10 -W 6:00 -G s_stud_infk -R "rusage[mem=1500,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=16000]" "module load gcc/8.2.0 python_gpu/3.8.5; pipenv run        python main.py --model v2GAN --data /cluster/scratch/zdavid/cil_data_root  --epochs 10000 --batch_size 8 --workers 10 --lr '$lr' \
     --save_dir save/ --img_dim 400 --h_flip --v_flip --brightness 0.1 --contrast 0.1 --rotate \
    --lmbda '$lambda' --train_data cil --val_data cil  --val_period 5  --beta1 0.5 --beta2 0.999 --loss '$l'"
      done

  done
done

