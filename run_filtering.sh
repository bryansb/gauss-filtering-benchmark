#!/bin/bash
sigma_values=(1 2 3 4)
kernel_sizes=(3 5 9 15)
#kernel_sizes=(3 5)

mkdir -p reports

echo "where,kernel_size,sigma_value,image_height,image_width,time" > reports/times_gpu.csv
echo "where,kernel_size,sigma_value,image_height,image_width,time" > reports/times_cpu.csv

echo "Iniciando ejecuciones en la GPU..."
for ks in "${kernel_sizes[@]}" 
do
    for sv in "${sigma_values[@]}"
    do
        name=$ks
        name+=$sv
        name+="_gpu.png"
        final_time=`python main.py -w gpu -ks $ks -sv $sv -in $name`
        echo "gpu,$ks,$sv,$final_time" >> reports/times_gpu.csv
    done
done

echo "Iniciando ejecuciones en la CPU..."
for ks in "${kernel_sizes[@]}" 
do
    for sv in "${sigma_values[@]}"
    do
        name=$ks
        name+=$sv
        name+="_cpu.png"
        final_time=`python main.py -w cpu -ks $ks -sv $sv -in $name`
        echo "cpu,$ks,$sv,$final_time" >> reports/times_cpu.csv
    done
done

exit 0
