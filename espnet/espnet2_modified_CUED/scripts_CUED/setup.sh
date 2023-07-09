cd /home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet
egs2/TEMPLATE/tts1/setup.sh egs2/CUED_vctk/tts_integrated
cd egs2/CUED_vctk/tts_integrated
cp ../../mini_an4/tts1/run.sh .
ln -s /home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/data_speaker_206/* .
cp ../../vctk/tts_gst/run.sh .
cp ../../vctk/tts_gst/run_grid.sh .
cp ../../vctk/tts_gst/submit_grid.sh .
cp ../../vctk/tts_gst/conf/* conf/ -r 
