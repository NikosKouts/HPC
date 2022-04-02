#!/bin/bash

THREADS=(1 2 4 8 16 32 64)


make

for thread in ${THREADS[@]}; do
  export OMP_NUM_THREADS=$thread
  #create directories for threads
  if [ ! -d "benchmarks/threads_$thread" ] 
  then
    mkdir "benchmarks/threads_$thread"
  fi
    #touch "benchmarks/threads_$thread/clusters_$cluster/$file/computation.txt" "benchmarks/threads_$thread/clusters_$cluster/$file/io.txt"

  for (( i=1; i<=12; i++ )) do  
    ./seq_main -i Image_data/texture17695.bin -n 2000 -o -b >> "benchmarks/threads_$thread/computation.txt" 2>> "benchmarks/threads_$thread/io.txt"
  done
done

make clean

