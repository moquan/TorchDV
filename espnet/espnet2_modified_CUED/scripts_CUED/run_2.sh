
Current working dir is /data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/espnet2_modified_CUED/scripts_CUED


cd /home/dawna/tts/mw545/tools/weblisteningtest 
python demotable_html_wav.py --idexp "^(p[0-9]+_[0-9]+)"  --tablename Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated \ 
/home/dawna/tts/mw545/tools/weblisteningtest/samples-mw545/2021_12_17_Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated/orig /home/dawna/tts/mw545/tools/weblisteningtest/samples-mw545/2021_12_17_Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated/cmp_2_stage /home/dawna/tts/mw545/tools/weblisteningtest/samples-mw545/2021_12_17_Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated/cmp_integrated \ 
--titles orig cmp_2_stage cmp_integrated 
rm -r /home/dawna/tts/mw545/tools/weblisteningtest/Samples_Webpage_Dir/2021_12_17_Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated 
mkdir /home/dawna/tts/mw545/tools/weblisteningtest/Samples_Webpage_Dir/2021_12_17_Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated 
mv Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated.html /home/dawna/tts/mw545/tools/weblisteningtest/Samples_Webpage_Dir/2021_12_17_Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated/ 


Add this line to /home/dawna/tts/mw545/tools/weblisteningtest/Samples_Webpage_Dir/2021_07_17_Voicebank_ESPNet2_Tacotron2.html: 
<li><a href="2021_12_17_Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated/Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated.html">Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated</a> </li>

Go back to /data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/espnet2_modified_CUED/scripts_CUED ? 

