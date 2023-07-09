echo Running on $HOSTNAME
export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
export WorkDir=debug_grid

export CUDA_HOME=/usr/local/cuda-10.1

# export TorchDVDir=/home/dawna/tts/mw545/TorchDV
# export VENV=py36torch15cuda101
# export MerlinDir=/home/dawna/tts/mw545/TorchDV/tools/merlin_cued_mw545
# export PATH=${PATH}:${CUDA_HOME}/bin:/home/dawna/tts/mw545/tools/anaconda2/bin
export TorchDVDir=/data/vectra2/tts/mw545/TorchDV
export VENV=espnetpy36torch15cuda101
export MerlinDir=/data/vectra2/tts/mw545/TorchDV/tools/merlin_cued_mw545
export PATH=${PATH}:${CUDA_HOME}/bin:/data/vectra2/tts/mw545/TorchTTS/anaconda/bin

cd ${TorchDVDir}/${WorkDir}
export PYTHONPATH=${PYTHONPATH}:${PWD}:${MerlinDir}:${MerlinDir}/tools/vocoders/straight:${MerlinDir}/tools/vocoders/pulsemodel:${MerlinDir}/tools/vocoders:${MerlinDir}/tools:${MerlinDir}/tools/sigproc

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
unset LD_PRELOAD

source activate ${VENV}
echo Working Directory $PWD
#OMP_NUM_THREADS=30 python ./test.py
OMP_NUM_THREADS=30 python ./run_24kHz.py ${PWD}
#python ./tests/pytorch_gpu_test.py
echo Finished "$(date +"%Y_%m_%d_%H_%M_%S")"

