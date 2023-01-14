#mv run_grid.sh.* logs/ 2>/dev/null
if [ $1 = cudaAll ]; then
    qsub -cwd -M mw545@cam.ac.uk -m e -S /bin/bash -o ${PWD} -e ${PWD} -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=* run_grid.sh $2 ${PWD}
elif [ $1 = cuda ]; then
    qsub -cwd -M mw545@cam.ac.uk -m e -S /bin/bash -o ${PWD} -e ${PWD} -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=volta run_grid.sh $2 ${PWD}
elif [ $1 = cpu ]; then
    qsub -cwd -M mw545@cam.ac.uk -m e -S /bin/bash -o ${PWD} -e ${PWD} -l queue_priority=low,tests=0,mem_grab=0M,osrel=* run_grid.sh $2 ${PWD}
else
    qsub -cwd -M mw545@cam.ac.uk -m e -S /bin/bash -o ${PWD} -e ${PWD} -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=air$1 run_grid.sh $2 ${PWD}
fi



# for h in 208 210 211 212 213; do
# qsub -cwd -M mw545@cam.ac.uk -m be -S /bin/bash -o ${PWD} -e ${PWD} -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=air${h} run_grid.sh
# done

# This block is to submit jobs to all GPU machines e.g. copy data to scratch
if false; then
  qsub -M mw545@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=air200 all_run_grid.sh
  qsub -M mw545@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=air201 all_run_grid.sh
  qsub -M mw545@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=air202 all_run_grid.sh
  qsub -M mw545@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=air203 all_run_grid.sh
  qsub -M mw545@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=air204 all_run_grid.sh
  qsub -M mw545@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=air205 all_run_grid.sh
  qsub -M mw545@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=air206 all_run_grid.sh
  qsub -M mw545@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=air207 all_run_grid.sh
  qsub -M mw545@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=air208 all_run_grid.sh
  qsub -M mw545@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=air209 all_run_grid.sh
fi
