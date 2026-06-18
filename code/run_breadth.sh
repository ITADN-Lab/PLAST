#!/bin/bash
cd /home/hadoop/workstation/md/AAAI/code
C10="/home/hadoop/workstation/md/TJU-V5(ATJU)-sourcecode/ATJU/dataset/cifar-10-batches-py"
RES=/home/hadoop/workstation/md/AAAI/results/cifar10_breadth
mkdir -p $RES
CM="--dataset cifar10 --classes_per_task 2 --n_tasks 40 --epochs_per_task 3 --n_per_task 4000 --width 32 --data $C10"
# 波1: 核心对比 n=3
for s in 0 1 2; do
  python3 plast_resnet.py --method ptc_muon --no_vreset --ptc_vreset 0.5 --reperturb_frac 0.05 $CM --seed $s --out $RES/muonRP_s$s.json > ../logs/br_muonRP_s$s.log 2>&1 &
  python3 plast_resnet.py --method muon $CM --seed $s --out $RES/muon_s$s.json > ../logs/br_muon_s$s.log 2>&1 &
  python3 plast_resnet.py --method cbp  $CM --seed $s --out $RES/cbp_s$s.json  > ../logs/br_cbp_s$s.log 2>&1 &
  python3 plast_resnet.py --method adam $CM --seed $s --out $RES/adam_s$s.json > ../logs/br_adam_s$s.log 2>&1 &
done
wait
# 波2: AdamO λ 扫 (找CIFAR-10最优) + AdamO+RP
for lam in 1e-3 1e-2 3e-2; do for s in 0 1 2; do
  python3 plast_resnet.py --method adamo --adamo_lambda $lam $CM --seed $s --out $RES/adamo_lam${lam}_s$s.json > ../logs/br_adamo_${lam}_s$s.log 2>&1 &
done; done
wait
for s in 0 1 2; do
  python3 plast_resnet.py --method ptc_adamo --no_vreset --adamo_lambda 3e-2 --reperturb_frac 0.05 $CM --seed $s --out $RES/adamoRP_s$s.json > ../logs/br_adamoRP_s$s.log 2>&1 &
done
# permuted Muon 补 n=5
RESP=/home/hadoop/workstation/md/AAAI/results/pmnist_matched
ARCH="--mode permute --hidden 512 --depth 4 --n_tasks 80 --epochs_per_task 1 --n_per_task 6000"
for s in 3 4; do
  python3 plast_mnist.py --method muon --lr 1e-4 $ARCH --seed $s --out $RESP/muon_lr1e-4_s$s.json > ../logs/br_pmuon_s$s.log 2>&1 &
done
wait
echo "BREADTH_DONE"
