#!/bin/bash
cd /home/hadoop/workstation/md/AAAI/code
C100=/home/hadoop/workstation/md/LafTJU-TII/experiments/dataset/cifar-100-python
RES=/home/hadoop/workstation/md/AAAI/results/fair_final
mkdir -p $RES
COMMON="--dataset cifar100 --classes_per_task 10 --n_tasks 25 --epochs_per_task 2 --n_per_task 5000 --width 32 --data $C100"
# 波1: AdamO(最优λ=1e-2)+重扰动 (=AdamO+CBP, 对等公平), n=3  [AdamO重显存, 限6并发]
for s in 0 1 2; do
  python3 plast_resnet.py --method ptc_adamo --no_vreset --adamo_lambda 1e-2 --reperturb_frac 0.05 $COMMON --seed $s \
    --out $RES/adamoRP_lam1e-2_s$s.json > ../logs/ff_adamoRP_s$s.log 2>&1 &
done
wait
# 波2: 确认AdamO λ峰值 (3e-2, 1e-1)
for lam in 3e-2 1e-1; do for s in 0 1 2; do
  python3 plast_resnet.py --method adamo --adamo_lambda $lam $COMMON --seed $s \
    --out $RES/adamo_lam${lam}_s$s.json > ../logs/ff_adamo_${lam}_s$s.log 2>&1 &
done; done
wait
echo "FAIR_FINAL_DONE"
