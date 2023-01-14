echo Running on $HOSTNAME

export WorkDir=debug_nausicaa

#export VENV=py36torch12nau
#export CUDA_HOME=/usr/local/cuda-9.2

export VENV=py36torch15cuda101
export CUDA_HOME=/usr/local/cuda-10.1
#export CUDA_VISIBLE_DEVICES=3
#export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}

#source ~/.bashrc
export MerlinDir=/home/dawna/tts/mw545/TorchDV/tools/merlin_cued_mw545

cd /home/dawna/tts/mw545/TorchDV/${WorkDir}
#export PYTHONPATH=${PYTHONPATH}:${PWD}:${PWD}/tools/vocoders/straight
export PYTHONPATH=${PYTHONPATH}:${PWD}
export PYTHONPATH=${PYTHONPATH}:${MerlinDir}:${MerlinDir}/tools/vocoders/straight:${MerlinDir}/tools/vocoders/pulsemodel:${MerlinDir}/tools/vocoders:${MerlinDir}/tools:${MerlinDir}/tools/sigproc

export PATH=${PATH}:${CUDA_HOME}/bin:/home/dawna/tts/mw545/tools/anaconda2/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
unset LD_PRELOAD

source activate ${VENV}
echo Working Directory $PWD
#OMP_NUM_THREADS=30 python ./test.py
OMP_NUM_THREADS=30 python ./run_nn_iv_batch_T4_DV.py ${PWD}
#python ./tests/pytorch_gpu_test.py
echo Finished "$(date +"%Y_%m_%d_%H_%M_%S")"
