echo Running on $HOSTNAME

# export WorkDir=debug_nausicaa
# export WorkDir=/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/espnet2_modified_CUED/scripts_CUED
export WorkDir=/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/scripts_CUED
#export VENV=py36torch12nau
#export CUDA_HOME=/usr/local/cuda-9.2

export VENV=py36torch15cuda101
export CUDA_HOME=/usr/local/cuda-10.1
#export CUDA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}

#source ~/.bashrc
export MerlinDir=/home/dawna/tts/mw545/TorchDV/tools/merlin_cued_mw545

# cd /home/dawna/tts/mw545/TorchDV/${WorkDir}
cd ${WorkDir}
#export PYTHONPATH=${PYTHONPATH}:${PWD}:${PWD}/tools/vocoders/straight
export PYTHONPATH=${PYTHONPATH}:${PWD}
export PYTHONPATH=${PYTHONPATH}:${MerlinDir}:${MerlinDir}/tools/vocoders/straight:${MerlinDir}/tools/vocoders/pulsemodel:${MerlinDir}/tools/vocoders:${MerlinDir}/tools:${MerlinDir}/tools/sigproc

export PATH=${PATH}:${CUDA_HOME}/bin:/home/dawna/tts/mw545/tools/anaconda2/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
unset LD_PRELOAD

source activate ${VENV}
echo Working Directory ${PWD}
python ./make_data.py
#bash ./run_all.sh
echo Finished "$(date +"%Y_%m_%d_%H_%M_%S")"
