#mv run_grid.sh.* log/ 2>/dev/null
#qsub -cwd -M mw545@cam.ac.uk -m e -S /bin/bash -o ${PWD} -e ${PWD} -l queue_priority=low,tests=0,mem_grab=0M,osrel=* run_grid.sh
#qsub -M mw545@cam.ac.uk -m e -S /bin/bash -o ${PWD} -e ${PWD} -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=*,hostname=!air208 run_grid.sh
qsub -cwd -M mw545@cam.ac.uk -m e -S /bin/bash -o ${PWD} -e ${PWD} -l queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*,gpuclass=* run_grid.sh

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
