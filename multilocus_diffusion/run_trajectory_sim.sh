#!/bin/bash
for i in {1..10}
do
    echo "Simulation $i"
    slim freq_trajectory_sim.slim > $i.part &
done
wait
echo "SLiMulation done"
awk '/1 0.0/,0' 1.part > two_locus_trajectory.csv

for i in {2..10}
do
    awk '/1 0.0/,0' $i.part >> two_locus_trajectory.csv
done
echo "Finished writing trajectories into a .csv file"
rm *.part