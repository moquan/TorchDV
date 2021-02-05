cp /home/dawna/tts/mw545/TorchDV/debug_nausicaa/*.sh .
# Remember to change workdir
# TODO: Better: write ${PWD} to run_nausicaa.sh directly


mkdir log
ln -s /data/vectra2/tts/mw545/Data/data_voicebank data
ln -s /home/dawna/tts/mw545/TorchDV/tools/merlin_cued_mw545/exp_mw545
ln -s /home/dawna/tts/mw545/TorchDV/tools/merlin_cued_mw545/run_nn_iv_batch_T4_DV.py

rm run_nn_iv_batch_T4_DV.py; cp /home/dawna/tts/mw545/TorchDV/tools/merlin_cued_mw545/run_nn_iv_batch_T4_DV.py .
