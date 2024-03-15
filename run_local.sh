


PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python main.py --model v2GAN --data /ssd/cil_data_root/  --epochs 10000 --batch_size 5 --workers 10 --lr 0.0003 \
     --save_dir save/ --img_dim 400 --h_flip --v_flip --brightness 0.1 --contrast 0.1 --rotate --distort \
    --lmbda 0.1 --train_data cil --val_data cil  --val_period 1  --beta1 0.5 --beta2 0.999
