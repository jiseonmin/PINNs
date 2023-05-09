#!/bin/bash
for i in {1..20}
do
    echo "Simulation $i"
    slim -d S1=-1e-5 -d S2=-1e-5 -d MU1=1e-8 -d MU2=1e-8 freq_trajectory_sim.slim > $i.part &
done
wait
echo "SLiMulation done"
awk '/1 0.0/,0' 1.part > two_locus_trajectory.csv

for i in {2..20}
do
    awk '/1 0.0/,0' $i.part >> two_locus_trajectory.csv
done
echo "Finished writing trajectories into a .csv file"
rm *.part