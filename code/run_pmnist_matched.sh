#!/bin/bash
# 匹配 AdamO(ICML'26) 的 MLP 协议: depth=4 hidden=512; Adam/AdamO lr=1e-4; permuted-MNIST
cd /home/hadoop/workstation/md/AAAI/code
RES=/home/hadoop/workstation/md/AAAI/results/pmnist_matched
LOG=/home/hadoop/workstation/md/AAAI/logs
mkdir -p $RES
ARCH="--mode permute --hidden 512 --depth 4 --n_tasks 80 --epochs_per_task 1 --n_per_task 6000"
run() { python3 plast_mnist.py "$@" >> $LOG/pmnist_matched.log 2>&1 & }
for s in 0 1 2 3 4; do
  # Adam 家族 lr=1e-4 (匹配AdamO)
  run --method adam  --lr 1e-4 $ARCH --seed $s --out $RES/adam_s$s.json
  run --method adamo --lr 1e-4 --adamo_lambda 1e-3 $ARCH --seed $s --out $RES/adamo_s$s.json
  run --method cbp   --lr 1e-4 $ARCH --seed $s --out $RES/cbp_s$s.json
  run --method ptc   --lr 1e-4 --ptc_vreset 0.1 --rank_target 350 $ARCH --seed $s --out $RES/ptc_s$s.json
  run --method ptc_adamo --lr 1e-4 --adamo_lambda 1e-3 --ptc_vreset 0.1 --rank_target 350 $ARCH --seed $s --out $RES/ptc_adamo_s$s.json
  wait  # 每个seed一波(~11 job), 控显存
done
# Muon LR 公平扫 (单独, 取最优后再ptc): lr ∈ {1e-4,3e-4,1e-3}
for lr in 1e-4 3e-4 1e-3; do
  for s in 0 1 2; do
    run --method muon     --lr $lr $ARCH --seed $s --out $RES/muon_lr${lr}_s$s.json
    run --method ptc_muon --lr $lr --ptc_vreset 0.1 --rank_target 350 $ARCH --seed $s --out $RES/ptc_muon_lr${lr}_s$s.json
  done
  wait
done
echo "PMNIST_MATCHED_DONE"
