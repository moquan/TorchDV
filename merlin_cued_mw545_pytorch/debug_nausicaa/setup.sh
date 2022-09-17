export WorkDir=[Name of workdir here]
mkdir ${WorkDir}
cd ${WorkDir}
cp /data/vectra2/tts/mw545/TorchDV/tools/template/*.sh .
sed -i "s/export WorkDir=debug_grid/export WorkDir=${WorkDir}/" run_grid.sh

export MerlinDir=/data/vectra2/tts/mw545/TorchDV/tools/merlin_cued_mw545
mkdir log
ln -s ${MerlinDir}/run_24kHz.py
ln -s ${MerlinDir}/config_24kHz.py
