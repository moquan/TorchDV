from run_nn_iv_batch_T4_DV import configuration
from frontend_mw545.data_io import Data_File_Directory_Utils

work_dir = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa'
cfg = configuration(work_dir)

data_list_writer = Data_File_Directory_Utils(cfg)
data_list_writer.copy_to_scratch()





'''



import os, sys
import shutil

dir_pair_list = []

# dir_pair_list.append(['/home/dawna/tts/mw545/TorchDV/debug_grid/data/nn_lab_resil_norm_601', '/scratch/tmp-mw545/voicebank_208_speakers/nn_lab_resil_norm_601'])
# dir_pair_list.append(['/home/dawna/tts/mw545/TorchDV/debug_grid/data/nn_cmp_resil_norm_86', '/scratch/tmp-mw545/voicebank_208_speakers/nn_cmp_resil_norm_86'])
dir_pair_list.append(['/home/dawna/tts/mw545/TorchDV/debug_grid/data/nn_wav_resil_norm_80', '/scratch/tmp-mw545/voicebank_208_speakers/nn_wav_resil_norm_80'])
# dir_pair_list.append(['/home/dawna/tts/mw545/Data/Data_Voicebank_48kHz_Pitch', '/scratch/tmp-mw545/voicebank_208_speakers/pitch'])
dir_pair_list.append(['/home/dawna/tts/mw545/TorchDV/debug_grid/data/nn_f016k_resil', '/scratch/tmp-mw545/voicebank_208_speakers/nn_f016k_resil'])
dir_pair_list.append(['/home/dawna/tts/mw545/TorchDV/debug_grid/data/nn_pitch_resil', '/scratch/tmp-mw545/voicebank_208_speakers/nn_pitch_resil'])

assert len(dir_pair_list) > 0

for dir_pair in dir_pair_list:
  ori_dir = dir_pair[0]
  tar_dir = dir_pair[1]

  try:
      shutil.rmtree(tar_dir)
      print("Removing target directory: "+tar_dir)
      sys.stdout.flush()
  except:
      print("Target directory does not exist yet: "+tar_dir)
      sys.stdout.flush()
  os.makedirs(tar_dir)

  file_list = os.listdir(ori_dir)
  for file_name in file_list:
    # if 'mu' not in file_name:
    x = os.path.join(ori_dir, file_name)
    y = os.path.join(tar_dir, file_name)
    shutil.copyfile(x, y)

'''