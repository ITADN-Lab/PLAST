#!/bin/bash
cd /home/hadoop/workstation/md/AAAI/code
C100=/home/hadoop/workstation/md/LafTJU-TII/experiments/dataset/cifar-100-python
RES=/home/hadoop/workstation/md/AAAI/results/width128_c100
mkdir -p $RES
CM="--dataset cifar100 --classes_per_task 10 --n_tasks 25 --epochs_per_task 2 --n_per_task 5000 --width 128 --data $C100"
for s in 0 1 2; do
  python3 plast_resnet.py --method ptc_muon --no_vreset --ptc_vreset 0.5 --reperturb_frac 0.05 $CM --seed $s --out $RES/muonRP_s$s.json > ../logs/w128_muonRP_s$s.log 2>&1 &
  python3 plast_resnet.py --method adamo --adamo_lambda 3e-2 $CM --seed $s --out $RES/adamo_s$s.json > ../logs/w128_adamo_s$s.log 2>&1 &
  wait
done
echo "W128_DONE"
