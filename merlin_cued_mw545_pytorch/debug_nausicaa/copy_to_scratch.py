import os
import shutil

ori_dir = '/home/dawna/tts/mw545/DVExp/debug/data/nn_cmp_resil_norm_86'
tar_dir = '/scratch/tmp-mw545/voicebank_208_speakers/nn_cmp_resil_norm_86'
#ori_dir = '/home/dawna/tts/mw545/Data/Data_Voicebank_48kHz_Pitch_Resil'
#tar_dir = '/scratch/tmp-mw545/voicebank_208_speakers/pitch'

try:
    shutil.rmtree(tar_dir)
    print "Removing target directory: "+tar_dir
except:
    print "Target directory does not exist yet: "+tar_dir
os.makedirs(tar_dir)

file_list = os.listdir(ori_dir)
for file_name in file_list:
  if 'mu' not in file_name:
    x = os.path.join(ori_dir, file_name)
    y = os.path.join(tar_dir, file_name)
  shutil.copyfile(x, y)
