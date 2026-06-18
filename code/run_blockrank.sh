#!/bin/bash
cd /home/hadoop/workstation/md/AAAI/code
C100=/home/hadoop/workstation/md/LafTJU-TII/experiments/dataset/cifar-100-python
RES=/home/hadoop/workstation/md/AAAI/results/blockrank
mkdir -p $RES
CM="--log_ntk --dataset cifar100 --classes_per_task 10 --n_tasks 25 --epochs_per_task 2 --n_per_task 5000 --width 32 --data $C100"
for s in 0 1 2; do
  python3 plast_resnet.py --method adam $CM --seed $s --out $RES/adam_s$s.json > ../logs/br_adam_s$s.log 2>&1 &
  python3 plast_resnet.py --method muon $CM --seed $s --out $RES/muon_s$s.json > ../logs/br_muon_s$s.log 2>&1 &
  python3 plast_resnet.py --method adamo --adamo_lambda 3e-2 $CM --seed $s --out $RES/adamo_s$s.json > ../logs/br_adamo_s$s.log 2>&1 &
  python3 plast_resnet.py --method ptc_muon --no_vreset --ptc_vreset 0.5 --reperturb_frac 0.05 $CM --seed $s --out $RES/muonRP_s$s.json > ../logs/br_muonRP_s$s.log 2>&1 &
  wait
done
echo "BLOCKRANK_DONE"
