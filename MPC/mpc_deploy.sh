#!/bin/bash

password="mpctest1!"
i=0
n=2
(cd MP-SPDZ && ./compile.py -R 64 end)&
for line in $(cat ip.txt)
do 
    (./expect.exp ${line} ${password} ${i} ${n})&
done
wait

(cd MP-SPDZ && ./spdz2k-party.x -N ${n} -p 0 -h 1.13.168.17  -F -OF END end )&# 
for line in $(cat ip.txt)
do  
    ((i++))
    (
    ./run.exp ${line} ${password} ${i} ${n})&
done
wait