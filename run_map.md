                

module load gcc/8.2.0 python_gpu/3.8.5
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train_YNet.py \
        --data /cluster/scratch/makleine/CIL/data/cil-road-segmentation --epochs 10000 --batch_size 16 --workers 2 \
        --lr 0.0001 --output_dir /cluster/scratch/makleine/CIL/output_dirs/test9 --img_dim 400 --img_channels 3

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)














224832294:
        --lr 0.0001 --momentum 0.9 --weight_decay 0.01
SGD
test12


224832458:
        --lr 0.0005 --momentum 0.9 --weight_decay 0.01
SGD
test13



224832510:
        --lr 0.0005 --momentum 0.9 --weight_decay 0.1
SGD
test14


224832769:
        --lr 0.0001 --momentum 0.9 --weight_decay 0.1                   FAIL!



224832972:
        --lr 0.0001 --weight_decay 0.01                                 FAIL!
AdamW   

224833039:
        --lr 0.0005 --weight_decay 0.01                                 FAIL!
AdamW   



224833124:
        --lr 0.00001 --weight_decay 0.001                                
AdamW   
test15
-----------------------------
224833174:
        --lr 0.00005 --weight_decay 0.0001                                
AdamW   
test16
--->>> also crashes after 67 epochs...
-----------------------------

224833831:
        --lr 0.0001 --weight_decay 0.0001                                
AdamW   
test17
--->>> WTF ERROR!!!

224834087:
        --lr 0.00005 --weight_decay 0.00001                                
AdamW   
test18


------
BELOW!!!!!!!!!!!!!!!!!!!!!!
bce loss with logits!!!!!!!!!!

224834588:
        --lr 0.00005 --weight_decay 0.0001                                
AdamW   
test19



224834712:
        --lr 0.0001 --weight_decay 0.0001                    FAIL!!!            
AdamW   
test20


224834776
        --lr 0.00005 --weight_decay 0.00001                                
AdamW   
test20


224834857:
        0.0001 --momentum 0.9 --weight_decay 0.0001

AdamW
test21








### Running
JOBID      USER    STAT  QUEUE      FROM_HOST   EXEC_HOST   JOB_NAME   SUBMIT_TIME
224832510  maklein RUN   gpuhe.4h   eu-login-08 5*eu-g3-084 *hannels 3 Jul  7 23:17
224833174  maklein RUN   gpuhe.4h   eu-login-08 5*eu-g3-055 *hannels 3 Jul  7 23:30
224832294  maklein RUN   gpuhe.4h   eu-login-08 5*eu-g3-083 *hannels 3 Jul  7 23:05
224832458  maklein RUN   gpuhe.4h   eu-login-08 5*eu-g3-062 *hannels 3 Jul  7 23:14
224833124  maklein RUN   gpuhe.4h   eu-login-08 5*eu-g3-060 *hannels 3 Jul  7 23:28

------





### lsf.o224909559                              --- did not really work
rm -rf ../output_dirs/new1

module load gcc/8.2.0 python_gpu/3.8.5
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train_YNet.py \
        --data /cluster/scratch/makleine/CIL/data/cil-road-segmentation --epochs 1000 --batch_size 16 --workers 2 \
        --lr 0.00005 --momentum 0.9 --weight_decay 0.00001 --output_dir /cluster/scratch/makleine/CIL/output_dirs/new1 \
        --img_dim 400 --img_channels 3 --ynet_version 2



### lsf.o224909549                              --- did not really work
rm -rf ../output_dirs/new2

module load gcc/8.2.0 python_gpu/3.8.5
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train_YNet.py \
        --data /cluster/scratch/makleine/CIL/data/cil-road-segmentation --epochs 1000 --batch_size 16 --workers 2 \
        --lr 0.00005 --momentum 0.9 --weight_decay 0.00001 --output_dir /cluster/scratch/makleine/CIL/output_dirs/new2 \
        --img_dim 400 --img_channels 3 --ynet_version 3




### lsf.o224913172                              -> FAIL
rm -rf ../output_dirs/new3

module load gcc/8.2.0 python_gpu/3.8.5
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096

python train_YNet.py \
--lr 0.0001 --momentum 0.9 --weight_decay 0.0001 --output_dir /cluster/scratch/makleine/CIL/output_dirs/new3 \
--ynet_version 3




### lsf.o224913326
        --lr 0.00001 --momentum 0.9 --weight_decay 0.0001 --ynet_version 3 
new3

### lsf.o224913895
        --lr 0.00001 --momentum 0.9 --weight_decay 0.0001 --ynet_version 2 
new4


### 224920277                           -> FAIL
        --lr 0.00001 --momentum 0.9 --weight_decay 0.000001 --ynet_version 3
new5


### 224920295                           -> FAIL
        --lr 0.00001 --momentum 0.9 --weight_decay 0.000001 --ynet_version 2
new6









