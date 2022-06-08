pip install neptune-contrib


python3 /home/dircon/yuje/CoBaL_code/main.py cifar10 -a resnet --layers 56 -C -g 0 --save DPFWBNR2layerimp105corr1e5 --run-type train \
-P --pruner dpf --prune-type structured --prune-freq 16 --prune-rate 0.3 --prune-imp group_coba --prune-imptype L2 \
--batch-size 128 --epochs 300 --lr 0.2 --wd 1e-4 --nesterov --scheduler multistep --milestones 150 225 --gamma 0.1 \
--bias_importance 0.01 --imp_type 7 -g 6