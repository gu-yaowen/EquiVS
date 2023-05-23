lr=0.001
wd=1e-4
ep=200
dp=0.05
bs=128
hs=128
nl=2
python main.py -id 3 -sp ${lr}_0_${ep}_${dp}_${bs}_${hs}_${nl} \
-ta regression -nj 50 -se 0 -lr ${lr} -ep ${ep} -dp ${dp} -hs ${hs} \
-bs ${bs} -wd ${wd} -nl ${nl} -cm R2 -mo train