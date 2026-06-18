#!/bin/bash
cd /home/hadoop/workstation/md/AAAI/code
C100=/home/hadoop/workstation/md/LafTJU-TII/experiments/dataset/cifar-100-python
RES=/home/hadoop/workstation/md/AAAI/results/causal_update
mkdir -p $RES
CM="--dataset cifar100 --classes_per_task 10 --n_tasks 25 --epochs_per_task 2 --n_per_task 5000 --width 32 --data $C100"
for rf in 0.05 0.1 0.25 0.5 1.0; do
  for s in 0 1 2; do
    python3 plast_resnet.py --method ptc_muon --no_vreset --ptc_vreset 0.5 --reperturb_frac 0.05 --muon_rank_frac $rf $CM --seed $s --out $RES/muonRP_rf${rf}_s$s.json > ../logs/cu_rf${rf}_s$s.log 2>&1 &
  done
  wait
done
echo "CAUSAL_UPDATE_DONE"
