#!/bin/bash
cd /home/hadoop/workstation/md/AAAI/code
C100=/home/hadoop/workstation/md/LafTJU-TII/experiments/dataset/cifar-100-python
RES=/home/hadoop/workstation/md/AAAI/results/causal_bottleneck
mkdir -p $RES
CM="--dataset cifar100 --classes_per_task 10 --n_tasks 25 --epochs_per_task 2 --n_per_task 5000 --width 32 --log_rank --data $C100"
for k in 8 16 32 0; do
  for s in 0 1 2; do
    python3 plast_resnet.py --method ptc_muon --no_vreset --ptc_vreset 0.5 --reperturb_frac 0.05 --bottleneck $k $CM --seed $s --out $RES/muonRP_k${k}_s$s.json > ../logs/cau_muonRP_k${k}_s$s.log 2>&1 &
    python3 plast_resnet.py --method adamo --adamo_lambda 3e-2 --bottleneck $k $CM --seed $s --out $RES/adamo_k${k}_s$s.json > ../logs/cau_adamo_k${k}_s$s.log 2>&1 &
  done
  wait
done
echo "CAUSAL_DONE"
