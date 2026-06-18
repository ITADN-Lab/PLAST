#!/bin/bash
cd /home/hadoop/workstation/md/AAAI/code
C100=/home/hadoop/workstation/md/LafTJU-TII/experiments/dataset/cifar-100-python
RES=/home/hadoop/workstation/md/AAAI/results/baselines_c100
COMMON="--dataset cifar100 --classes_per_task 10 --n_tasks 25 --epochs_per_task 2 --n_per_task 5000 --width 32 --data $C100"
for s in 0 1 2 3 4; do
  for m in sgd adamw l2init redo shrink_perturb; do
    python3 plast_resnet.py --method $m $COMMON --seed $s --out $RES/${m}_s$s.json > ../logs/bl_${m}_s$s.log 2>&1 &
  done
  wait
done
echo "BASELINES_DONE"
