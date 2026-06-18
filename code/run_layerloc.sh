#!/bin/bash
cd /home/hadoop/workstation/md/AAAI/code
C100=/home/hadoop/workstation/md/LafTJU-TII/experiments/dataset/cifar-100-python
RES=/home/hadoop/workstation/md/AAAI/results/layerloc
mkdir -p $RES
CM="--method ptc_muon --reperturb_frac 0.05 --dataset cifar100 --classes_per_task 10 --n_tasks 25 --epochs_per_task 2 --n_per_task 5000 --width 32 --data $C100"
for s in 0 1 2; do
  for b in early mid late all none; do
    python3 plast_resnet.py $CM --muon_blocks $b --seed $s --out $RES/${b}_s$s.json > ../logs/loc_${b}_s$s.log 2>&1 &
  done
  wait
done
echo "LAYERLOC_DONE"
