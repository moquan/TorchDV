echo Running on $HOSTNAME, X_SGE_CUDA_DEVICE ${X_SGE_CUDA_DEVICE}
echo Loading $1
export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
export workDir=$2

export CUDA_HOME=/usr/local/cuda-10.1

# export VENV=py36torch15cuda101
# export PATH=${PATH}:${CUDA_HOME}/bin:/home/dawna/tts/mw545/tools/anaconda2/bin
export VENV=espnetpy36torch15cuda101
export MerlinDir=${workDir}/../merlin_cued_mw545_pytorch
export PATH=${PATH}:${CUDA_HOME}/bin:/data/vectra2/tts/mw545/TorchTTS/anaconda/bin

cd ${workDir}
export PYTHONPATH=${PYTHONPATH}:${PWD}:${MerlinDir}
# export PYTHONPATH=${PYTHONPATH}:${MerlinDir}/tools/vocoders/straight:${MerlinDir}/tools/vocoders/pulsemodel:${MerlinDir}/tools/vocoders:${MerlinDir}/tools:${MerlinDir}/tools/sigproc

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
unset LD_PRELOAD

source activate ${VENV}
echo Working Directory $PWD
OMP_NUM_THREADS=30 python ${MerlinDir}/run_24kHz.py ${workDir} $1

#OMP_NUM_THREADS=30 python ./test.py
#python ./tests/pytorch_gpu_test.py
echo Finished "$(date +"%Y_%m_%d_%H_%M_%S")"
