#!/bin/bash
cd /home/hadoop/workstation/md/AAAI/code
C100=/home/hadoop/workstation/md/LafTJU-TII/experiments/dataset/cifar-100-python
COMMON="--dataset cifar100 --classes_per_task 10 --n_tasks 25 --epochs_per_task 2 --n_per_task 5000 --width 32 --data $C100"
# 波1: muon+重扰动 补 seed3,4 (→n=5); AdamO最优λ=3e-2 + 重扰动 n=5 (对等公平最终版)
RES=/home/hadoop/workstation/md/AAAI/results/ablation_muon
RES2=/home/hadoop/workstation/md/AAAI/results/fair_final
for s in 3 4; do
  python3 plast_resnet.py --method ptc_muon --no_vreset --ptc_vreset 0.5 --reperturb_frac 0.05 $COMMON --seed $s \
    --out $RES/reperturb_only_s$s.json > ../logs/sol_muonRP_s$s.log 2>&1 &
done
for s in 0 1 2; do
  python3 plast_resnet.py --method ptc_adamo --no_vreset --adamo_lambda 3e-2 --reperturb_frac 0.05 $COMMON --seed $s \
    --out $RES2/adamoRP_lam3e-2_s$s.json > ../logs/sol_adamoRP_s$s.log 2>&1 &
done
wait
# 波2: AdamO最优λ=3e-2 + AdamO+RP 各补 seed3,4 → n=5; 机理rank for AdamO λ=3e-2
RES3=/home/hadoop/workstation/md/AAAI/results/mechanism_rank
for s in 3 4; do
  python3 plast_resnet.py --method ptc_adamo --no_vreset --adamo_lambda 3e-2 --reperturb_frac 0.05 $COMMON --seed $s \
    --out $RES2/adamoRP_lam3e-2_s$s.json > ../logs/sol_adamoRP_s$s.log 2>&1 &
done
python3 plast_resnet.py --method adamo --adamo_lambda 3e-2 --log_rank $COMMON --seed 0 --out $RES3/adamoBest_s0.json > ../logs/sol_rkadamo_s0.log 2>&1 &
python3 plast_resnet.py --method adamo --adamo_lambda 3e-2 --log_rank $COMMON --seed 1 --out $RES3/adamoBest_s1.json > ../logs/sol_rkadamo_s1.log 2>&1 &
wait
echo "SOLIDIFY_DONE"
