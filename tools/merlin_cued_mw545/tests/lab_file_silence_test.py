# lab_file_silence_test.py
# This script calculates amount of silence in each file
import os
from modules import read_file_list
from frontend.silence_reducer_keep_sil import SilenceReducer

file_id_list_file = '/home/dawna/tts/mw545/TorchDV/file_id_list.scp'

file_id_list = read_file_list(file_id_list_file)

lab_dir = '/data/vectra2/tts/mw545/Data/Data_Voicebank_48kHz_PML/label_state_align'
silence_reducer = SilenceReducer(n_cmp = 1)

num_sil_dict = {}
sil_index_file = '/home/dawna/tts/mw545/TorchDV/sil_index_list.scp'
f = open(sil_index_file, 'w')

for file_id in file_id_list:
    lab_file_name = os.path.join(lab_dir, file_id+'.lab')
    nonsilence_indices_200 = silence_reducer.load_alignment(lab_file_name)
    num_sil = nonsilence_indices_200[0]
    try: num_sil_dict[num_sil] += 1
    except: num_sil_dict[num_sil] = 1
    str_to_write = '%s %i %i' %(file_id, nonsilence_indices_200[0], nonsilence_indices_200[-1])
    f.write(str_to_write+'\n')

f.close()

print(num_sil_dict)
print(num_sil_dict.keys())


num_sil_dict = {}
sil_index_file = '/home/dawna/tts/mw545/TorchDV/sil_index_list.scp'
with open(sil_index_file, 'r') as f:
    file_lines = f.readlines()

for l in file_lines:
    a,b,c = l.strip().split(' ')
    file_id = a
    index_first = int(b)
    index_last  = int(c)
    num_sil_dict[file_id] = [index_first, index_last]
