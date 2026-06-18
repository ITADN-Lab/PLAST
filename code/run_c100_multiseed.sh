#!/bin/bash
cd /home/hadoop/workstation/md/AAAI/code
C100=/home/hadoop/workstation/md/LafTJU-TII/experiments/dataset/cifar-100-python
RES=/home/hadoop/workstation/md/AAAI/results/p0_gate
LOG=/home/hadoop/workstation/md/AAAI/logs
COMMON="--dataset cifar100 --classes_per_task 10 --n_tasks 25 --epochs_per_task 2 --n_per_task 5000 --width 32 --data $C100"
launch() { # seed
  local s=$1
  for m in adam muon adamo ptc ptc_muon ptc_adamo; do
    extra=""; case $m in ptc|ptc_adamo|ptc_muon) extra="--ptc_vreset 0.5 --reperturb_frac 0.05";; esac
    python3 plast_resnet.py --method $m $COMMON --seed $s $extra \
      --out $RES/${m}_s${s}.json > $LOG/p0_${m}_s${s}.log 2>&1 &
  done
}
# 波1: seeds 1,2 (12 jobs)
launch 1; launch 2; wait
# 波2: seeds 3,4
launch 3; launch 4; wait
echo "ALL_MULTISEED_DONE"
