#!/bin/bash
# Muon LR 公平性: 裸muon与ptc_muon在多个lr下, 证明ptc_muon的优势非来自裸muon欠调
cd /home/hadoop/workstation/md/AAAI/code
C100=/home/hadoop/workstation/md/LafTJU-TII/experiments/dataset/cifar-100-python
RES=/home/hadoop/workstation/md/AAAI/results/muon_lr_c100
mkdir -p $RES
COMMON="--dataset cifar100 --classes_per_task 10 --n_tasks 25 --epochs_per_task 2 --n_per_task 5000 --width 32 --data $C100"
for lr in 3e-4 1e-3 3e-3; do
  for s in 0 1 2; do
    nohup python3 plast_resnet.py --method muon --lr $lr $COMMON --seed $s \
      --out $RES/muon_lr${lr}_s$s.json > ../logs/mlr_muon_${lr}_s$s.log 2>&1 &
    nohup python3 plast_resnet.py --method ptc_muon --lr $lr --ptc_vreset 0.5 --reperturb_frac 0.05 $COMMON --seed $s \
      --out $RES/ptcmuon_lr${lr}_s$s.json > ../logs/mlr_ptcmuon_${lr}_s$s.log 2>&1 &
  done
  wait
done
echo "MUON_LR_C100_DONE"
