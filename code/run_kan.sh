#!/bin/bash
cd /home/hadoop/workstation/md/AAAI/code
RES=/home/hadoop/workstation/md/AAAI/results/kan_pilot
for s in 0 1 2 3 4; do
  python3 plast_kan.py --arch kan --hidden 64  --n_tasks 40 --n_per_task 4000 --seed $s --out $RES/kan_s$s.json   > ../logs/kan_kan_s$s.log 2>&1 &
  python3 plast_kan.py --arch mlp --hidden 256 --n_tasks 40 --n_per_task 4000 --seed $s --out $RES/mlp_s$s.json   > ../logs/kan_mlp_s$s.log 2>&1 &
  python3 plast_kan.py --arch mlp --hidden 64  --n_tasks 40 --n_per_task 4000 --seed $s --out $RES/mlpS_s$s.json  > ../logs/kan_mlpS_s$s.log 2>&1 &
done
wait
echo "KAN_PILOT_DONE"
