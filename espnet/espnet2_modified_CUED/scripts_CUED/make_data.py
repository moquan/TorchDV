######################################################
# make_data.py
# Data Prep for ESPNet 2 Tacotron 2 + CUED Voicebank #
# Note:                                              #
#   Some functions here write bash scripts to run    #
#   Run or submit after writing                      #
######################################################

import os, pickle, numpy, scipy, scipy.spatial, shutil, copy
import subprocess
from frontend_mw545.modules import prepare_script_file_path, read_file_list, File_List_Selecter, List_Random_Loader, Graph_Plotting, make_logger
from frontend_mw545.data_io import Data_Meta_List_File_IO

class dv_y_configuration(object):
  
    def __init__(self):
        self.espnet_root = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet'
        self.work_dir    = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk'
        self.raw_data_dir = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus' # currently: filelists  mel24  spk_embedding  txt  wav24  wav48
        self.file_id_list_dir = os.path.join(self.raw_data_dir, 'filelists/train_valid_test_SR')

        # self.orig_wav_dir = '/home/dawna/tts/import/VoiceBank/database/wav/Eng'
        self.orig_wav_dir = '/home/dawna/tts/gad27/features24_noforcedbin_nopulsecorr_VoiceBank/usedwav/Eng'
        self.orig_txt_dir = '/home/dawna/tts/import/VoiceBank/database/txt/Eng'

        self.data_dirs_to_link = os.path.join(self.work_dir, 'data_speaker_206')

        self.make_speaker_id_list_dict()
        self.get_spk_file_list()

    def get_spk_file_list(self):
        '''
        self.spk_file_list: keys are speakers, each value is a list
        '''
        # file_list_scp = '/home/dawna/tts/mw545/TorchDV/file_id_lists/file_id_list_used_cfg.scp'
        file_list_scp  = os.path.join(self.raw_data_dir, 'filelists/file_id_list_used_cfg.scp')
        self.spk_file_list = {}
        
        f_list = read_file_list(file_list_scp)
        for x in f_list:
            spk_id = x.split('_')[0]    
            try:
                self.spk_file_list[spk_id].append(x)
            except:
                self.spk_file_list[spk_id] = [x]

    def make_speaker_id_list_dict(self):
        speaker_id_list_dict = {}
        speaker_id_list_dict['all'] = ['p001', 'p002', 'p003', 'p004', 'p005', 'p006', 'p007', 'p008', 'p010', 'p011', 'p013', 'p014', 'p015', 'p017', 'p019', 'p021', 'p022', 'p023', 'p024', 'p026', 'p027', 'p028', 'p030', 'p031', 'p032', 'p033', 'p034', 'p035', 'p036', 'p037', 'p038', 'p039', 'p043', 'p044', 'p045', 'p047', 'p048', 'p049', 'p052', 'p053', 'p054', 'p055', 'p056', 'p057', 'p060', 'p061', 'p062', 'p063', 'p065', 'p067', 'p068', 'p069', 'p070', 'p071', 'p073', 'p074', 'p075', 'p076', 'p077', 'p079', 'p081', 'p084', 'p085', 'p087', 'p088', 'p089', 'p090', 'p091', 'p093', 'p094', 'p095', 'p096', 'p097', 'p098', 'p099', 'p100', 'p101', 'p102', 'p103', 'p105', 'p106', 'p107', 'p109', 'p110', 'p112', 'p113', 'p114', 'p116', 'p117', 'p118', 'p120', 'p122', 'p123', 'p124', 'p125', 'p126', 'p128', 'p129', 'p130', 'p131', 'p132', 'p134', 'p135', 'p136', 'p139', 'p140', 'p141', 'p142', 'p146', 'p147', 'p151', 'p152', 'p153', 'p155', 'p156', 'p157', 'p158', 'p160', 'p161', 'p162', 'p163', 'p164', 'p165', 'p166', 'p167', 'p168', 'p170', 'p171', 'p173', 'p174', 'p175', 'p176', 'p177', 'p178', 'p179', 'p180', 'p182', 'p184', 'p187', 'p188', 'p192', 'p194', 'p197', 'p200', 'p201', 'p207', 'p208', 'p209', 'p210', 'p211', 'p212', 'p215', 'p216', 'p217', 'p218', 'p219', 'p220', 'p221', 'p223', 'p224', 'p290', 'p293', 'p294', 'p295', 'p298', 'p299', 'p300', 'p302', 'p303', 'p304', 'p306', 'p308', 'p310', 'p311', 'p312', 'p313', 'p314', 'p316', 'p320', 'p321', 'p322', 'p327', 'p331', 'p333', 'p334', 'p336', 'p337', 'p339', 'p340', 'p341', 'p343', 'p344', 'p347', 'p348', 'p349', 'p350', 'p351', 'p353', 'p354', 'p356', 'p370', 'p375', 'p376', 'p384', 'p386', 'p398']
        # p41 has been removed, voice of a sick person
        # p202 is not in file_id_list yet, and he has same voice as p209, be careful
        speaker_id_list_dict['valid'] = ['p162', 'p002', 'p303', 'p048', 'p109', 'p153', 'p038', 'p166', 'p218', 'p070']    # Last 3 are males
        speaker_id_list_dict['test']  = ['p293', 'p210', 'p026', 'p024', 'p313', 'p223', 'p141', 'p386', 'p178', 'p290'] # Last 3 are males
        speaker_id_list_dict['not_train'] = speaker_id_list_dict['valid']+speaker_id_list_dict['test']
        speaker_id_list_dict['train'] = [spk for spk in speaker_id_list_dict['all'] if (spk not in speaker_id_list_dict['not_train'])]

        for k in speaker_id_list_dict:
            speaker_id_list_dict[k].sort()
        self.speaker_id_list_dict = speaker_id_list_dict

def pad_speaker_id(file_id, l_id=4):
    # Pad speaker id with 0s, so lengths are all l_id
    speaker_id = file_id.split('_')[0]
    l = len(speaker_id)
    if l < l_id:
        # Pad zeros
        file_id_new = 'p' + (l_id-l)*'0' + file_id[1:]
        return file_id_new
    else:
        return file_id

def unpad_speaker_id(file_id):
    # Remove the padded 0s in speaker id
    new_file_id = file_id
    while new_file_id[1] == '0':
        new_file_id = new_file_id[0] + new_file_id[2:]
    return new_file_id

class Make_Corpus(object):
    """docstring for Make_Corpus"""
    def __init__(self, dv_y_cfg):
        super(Make_Corpus, self).__init__()
        self.dv_y_cfg = dv_y_cfg

    def run(self):
        # self.make_file_id_scp()
        # self.make_data_p320(run_alone=True)
        # self.make_corpus_dir()
        # self.make_wav24()
        self.make_wav_test_original()

        # self.write_submit_file()

    def make_file_id_scp(self):
        orig_file_list_scp = '/home/dawna/tts/mw545/TorchDV/file_id_lists/file_id_list_used_cfg.scp'
        new_file_list_scp  = os.path.join(self.dv_y_cfg.raw_data_dir, 'filelists/file_id_list_used_cfg.scp')

        orig_file_list = read_file_list(orig_file_list_scp)
        new_file_list  = [pad_speaker_id(x) for x in orig_file_list]
        new_file_list.sort()

        filelist_file = new_file_list_scp
        print('write to %s' % filelist_file)
        with open(filelist_file,'w') as f:
            for file_id in new_file_list:
                f.write(file_id+'\n')

    def make_corpus_dir(self):
        wav_dir = os.path.join(self.dv_y_cfg.raw_data_dir, 'wav48')
        txt_dir = os.path.join(self.dv_y_cfg.raw_data_dir, 'txt')

        for speaker_id in self.dv_y_cfg.speaker_id_list_dict['all']:
            # Handle p320 separately
            if speaker_id == 'p320':
                self.make_data_p320()
            else:
                target_wav_dir = os.path.join(wav_dir, speaker_id)
                target_txt_dir = os.path.join(txt_dir, speaker_id)
                prepare_script_file_path(target_wav_dir)
                prepare_script_file_path(target_txt_dir)
                file_id_list = self.dv_y_cfg.spk_file_list[speaker_id]

                orig_speaker_id = unpad_speaker_id(speaker_id)
                orig_wav_dir = os.path.join(self.dv_y_cfg.orig_wav_dir, orig_speaker_id)
                orig_txt_dir = os.path.join(self.dv_y_cfg.orig_txt_dir, orig_speaker_id)

                write_file_name = 'bash_dir/%s.sh' % speaker_id
                print('Writing to %s' % write_file_name)
                with open(write_file_name, 'w') as f:
                    for file_id in file_id_list:
                        orig_file_id = unpad_speaker_id(file_id)
                        orig_wav_file = os.path.join(orig_wav_dir, orig_file_id+'.used.wav')
                        orig_txt_file = os.path.join(orig_txt_dir, orig_file_id+'.txt')
                        new_wav_file  = os.path.join(target_wav_dir, file_id+'.wav')
                        new_txt_file  = os.path.join(target_txt_dir, file_id+'.txt')

                        s_write = 'ln -s %s %s\n' % (orig_wav_file, new_wav_file)
                        f.write(s_write)
                        s_write = 'ln -s %s %s\n' % (orig_txt_file, new_txt_file)
                        f.write(s_write)

        # Write a bash to run all speaker_id.sh
        write_file_name = 'run_all.sh'
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for speaker_id in self.dv_y_cfg.speaker_id_list_dict['all']:
                f.write('echo running %s.sh\n' % speaker_id)
                f.write('bash ./%s.sh\n' % speaker_id)

        print('Change run_grid.sh, bash ./%s' % write_file_name)
        print('Run submit_grid.sh')

    def make_data_p320(self, run_alone=False):
        # Special speaker: 2 folders in original data; p320, p320b
        # Use p320_[1...140] and p320b_[141...]
        # Link wav and text
        orig_wav_p320  = '/home/dawna/tts/gad27/features24_noforcedbin_nopulsecorr_VoiceBank/usedwav/Eng/p320'
        wav_p320_list  = os.listdir(orig_wav_p320)
        orig_wav_p320b = '/home/dawna/tts/gad27/features24_noforcedbin_nopulsecorr_VoiceBank/usedwav/Eng/p320b'
        wav_p320b_list = os.listdir(orig_wav_p320b)
        orig_txt_p320  = '/home/dawna/tts/import/VoiceBank/database/txt/Eng/p320'
        txt_p320_list  = os.listdir(orig_txt_p320)
        orig_txt_p320b = '/home/dawna/tts/import/VoiceBank/database/txt/Eng/p320b'
        txt_p320b_list = os.listdir(orig_txt_p320b)

        new_wav = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/wav48/p320'
        new_txt = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/txt/p320'
        new_wav  = os.path.join(self.dv_y_cfg.raw_data_dir, 'wav48/p320')
        new_txt  = os.path.join(self.dv_y_cfg.raw_data_dir, 'txt/p320')
        prepare_script_file_path(new_wav)
        prepare_script_file_path(new_txt)

        write_file_name = 'p320.sh'
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            if True:
                # Link <141 wav
                for file_name in wav_p320_list:
                    file_number = int(file_name.split('.')[0].split('_')[-1])
                    if file_number < 141:
                        s_write = 'ln -s %s/%s %s/%s\n' % (orig_wav_p320, file_name, new_wav, file_name.replace('used.wav','wav'))
                        f.write(s_write)

                # Link >140 wav
                # Actually, link all files in p320b
                for file_name in wav_p320b_list:
                    s_write = 'ln -s %s/%s %s/%s\n' % (orig_wav_p320b, file_name, new_wav, file_name.replace('used.wav','wav').replace('p320b','p320'))
                    f.write(s_write)

            if True:
                # Link <141 txt
                for file_name in txt_p320_list:
                    file_number = int(file_name.split('.')[0].split('_')[-1])
                    if file_number < 141:
                        s_write = 'ln -s %s/%s %s/%s\n' % (orig_txt_p320, file_name, new_txt, file_name)
                        f.write(s_write)

                # Link >140 txt
                # Actually, link all files in p320b
                for file_name in txt_p320b_list:
                    s_write = 'ln -s %s/%s %s/%s\n' % (orig_txt_p320b, file_name, new_txt, file_name.replace('p320b','p320'))
                    f.write(s_write)

        if run_alone:
            print('Run %s' % write_file_name)

    def link_wav24(self):
        '''
        DO NOT USE
        The wav24 here is wrong; wav24 generated from original wav48, before silence reduction
        Correct wav48 is /home/dawna/tts/gad27/features24_noforcedbin_nopulsecorr_VoiceBank/usedwav/Eng
        This matches label_state_align
        '''
        orig_wav24_dir = '/data/vectra2/tts/mw545/TorchTTS/VCTK-Corpus/wav24'
        new_wav24_dir  = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/wav24'

        for speaker_id in self.dv_y_cfg.speaker_id_list_dict['all']:
            target_wav_dir = os.path.join(new_wav24_dir, speaker_id)
            prepare_script_file_path(target_wav_dir)
            file_id_list = self.dv_y_cfg.spk_file_list[speaker_id]

            orig_speaker_id = unpad_speaker_id(speaker_id)
            orig_wav_dir = os.path.join(orig_wav24_dir, orig_speaker_id)

            write_file_name = '%s.sh' % speaker_id
            print('Writing to %s' % write_file_name)
            with open(write_file_name, 'w') as f:
                for file_id in file_id_list:
                    orig_file_id = unpad_speaker_id(file_id)
                    orig_wav_file = os.path.join(orig_wav_dir, orig_file_id+'.wav')
                    new_wav_file  = os.path.join(target_wav_dir, file_id+'.wav')

                    s_write = 'ln -s %s %s\n' % (orig_wav_file, new_wav_file)
                    f.write(s_write)

    def make_wav24(self):
        wav48_dir = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/wav48'
        wav24_dir = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/wav24'

        Sox = '/home/dawna/tts/mw545/tools/sox-14.4.1/src/sox'
        bash_dir = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/espnet2_modified_CUED/scripts_CUED/bash_dir'

        speaker_id_list = os.listdir(wav48_dir)

        for speaker_id in speaker_id_list:
            old_wav_dir = os.path.join(wav48_dir, speaker_id)
            new_wav_dir = os.path.join(wav24_dir, speaker_id)
            prepare_script_file_path(new_wav_dir)

            file_list = os.listdir(old_wav_dir)
            bash_file = os.path.join(bash_dir, speaker_id+'.sh')

            with open(bash_file, 'w') as f:
                for file_name in file_list:
                    old_file_name = os.path.join(old_wav_dir, file_name)
                    new_file_name = os.path.join(new_wav_dir, file_name)
                    f.write(Sox+' '+old_file_name+' -r 24000 -b 16 -c 1 '+new_file_name+'\n')

    def make_wav_test_original(self):
        '''
        Make a directory containing the original waveform samples, for comparison
        Link the files:
            1. directly in the new directory, instead of split by speakers
            2. new name, append '_gen'
        '''
        file_list_scp = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/filelists/train_valid_test_SR/test.scp'
        wav24_dir     = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/wav24'
        output_dir    = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/wav_ref'

        bash_dir = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/espnet2_modified_CUED/scripts_CUED/bash_dir'
        bash_file = os.path.join(bash_dir, 'run.sh')

        prepare_script_file_path(output_dir)

        file_id_list = read_file_list(file_list_scp)
        with open(bash_file, 'w') as f:
            for file_id in file_id_list:
                speaker_id = file_id.split('_')[0]
                orig_file_name = os.path.join(wav24_dir, speaker_id, file_id+'.wav')
                out_file_name  = os.path.join(output_dir, file_id+'_gen.wav')
                f.write('ln -s %s %s \n' % (orig_file_name, out_file_name))

    def write_submit_file(self):
        bash_dir = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/espnet2_modified_CUED/scripts_CUED/bash_dir'
        bash_file_list = os.listdir(bash_dir)
        submit_file = os.path.join(bash_dir, 'submit.sh')
        with open(submit_file, 'w') as f:
            for bash_file in bash_file_list:
                if (bash_file.split('.')[-1] == 'sh') and (bash_file != 'submit.sh'):
                    f.write('qsub -cwd -S /bin/bash -o ${PWD} -e ${PWD} -l queue_priority=low,tests=0,mem_grab=0M,osrel=*,gpuclass=* %s \n'%bash_file)

def write_train_valid_test_file_id_list(dv_y_cfg):
    '''
    Make 6 lists
    train, train_sr, valid, valid_sr, test, test_sr
    '''

    file_id_list_dir = dv_y_cfg.file_id_list_dir
    prepare_script_file_path(file_id_list_dir)

    filelist_file = os.path.join(file_id_list_dir,'tr_no_dev.scp')
    print('write to %s' % filelist_file)
    with open(filelist_file,'w') as f:
        for speaker_id in dv_y_cfg.speaker_id_list_dict['train']:
            for file_id in dv_y_cfg.spk_file_list[speaker_id]:
                file_number = int(file_id.split('_')[1])
                if int(file_number) > 80:
                    f.write(file_id+'\n')

    filelist_file = os.path.join(file_id_list_dir,'valid.scp')
    print('write to %s' % filelist_file)
    with open(filelist_file,'w') as f:
        for speaker_id in dv_y_cfg.speaker_id_list_dict['valid']:
            for file_id in dv_y_cfg.spk_file_list[speaker_id]:
                file_number = int(file_id.split('_')[1])
                if int(file_number) < 41:
                    f.write(file_id+'\n')

    filelist_file = os.path.join(file_id_list_dir,'test.scp')
    print('write to %s' % filelist_file)
    with open(filelist_file,'w') as f:
        for speaker_id in dv_y_cfg.speaker_id_list_dict['test']:
            for file_id in dv_y_cfg.spk_file_list[speaker_id]:
                file_number = int(file_id.split('_')[1])
                if int(file_number) < 41:
                    f.write(file_id+'\n')

    filelist_file = os.path.join(file_id_list_dir,'valid_SR.scp')
    print('write to %s' % filelist_file)
    with open(filelist_file,'w') as f:
        for speaker_id in dv_y_cfg.speaker_id_list_dict['valid']:
            for file_id in dv_y_cfg.spk_file_list[speaker_id]:
                file_number = int(file_id.split('_')[1])
                if int(file_number) < 81:
                    if int(file_number) > 40:
                        f.write(file_id+'\n')

    filelist_file = os.path.join(file_id_list_dir,'test_SR.scp')
    print('write to %s' % filelist_file)
    with open(filelist_file,'w') as f:
        for speaker_id in dv_y_cfg.speaker_id_list_dict['test']:
            for file_id in dv_y_cfg.spk_file_list[speaker_id]:
                file_number = int(file_id.split('_')[1])
                if int(file_number) < 81:
                    if int(file_number) > 40:
                        f.write(file_id+'\n')

class Make_Data(object):
    '''
    Make directory named "data"
    data
      |-- tr_no_dev
            segments  spk2utt  text  utt2spk  wav.scp
      |-- dev
      |-- eval1
    '''
    def __init__(self, dv_y_cfg):
        super(Make_Data, self).__init__()
        self.dv_y_cfg = dv_y_cfg

        self.data_dir = os.path.join(self.dv_y_cfg.data_dirs_to_link, 'data')
        prepare_script_file_path(self.data_dir)
        
        self.DMLFIO = Data_Meta_List_File_IO()
        self.file_id_list_num_sil_frame = self.DMLFIO.read_file_list_num_silence_frame()

        self.FLS = File_List_Selecter()


    def run(self):
        # for dir_name in ['tr_no_dev', 'dev', 'eval1']:
        for dir_name in ['listening_tests']:
            full_dir_name = os.path.join(self.data_dir, dir_name)
            prepare_script_file_path(full_dir_name)

            if dir_name == 'tr_no_dev':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'tr_no_dev.scp')
            if dir_name == 'dev':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'valid.scp')
            if dir_name == 'eval1':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'test.scp')
            if dir_name == 'listening_tests':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'listening_tests.scp')

            self.make_segments(full_dir_name, file_id_list_file)
            self.make_spk2utt(full_dir_name, file_id_list_file)
            self.make_text(full_dir_name, file_id_list_file)
            self.make_utt2spk(full_dir_name, file_id_list_file)
            self.make_wav_scp(full_dir_name, file_id_list_file)

    def sort_by_speaker_id(self, file_id):
        return file_id.split('_')[0]

    def make_segments(self, full_dir_name, file_id_list_file):
        '''
        Write file named "segments"
        segments:
        p225_001 p225_001 0.2850000 1.7600000
        mono-lab file p225_001:
            850000    2850000 pau
            2850000    3600000 p
            ...
            16050000   17600000 ax
            17600000   20400000 pau
        I have a file with silence indices, in 200Hz frames
        /home/dawna/tts/mw545/TorchDV/file_id_lists/data_meta/file_id_list_num_sil_frame.scp
            file_id_list_num_sil_frame['p100_041'] = [479, 102, 404]
        state-align file p100_041:
        /home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/label_state_align/p100_041.lab
            5050000 5100000 xx~xx-#+w=@:...
            ...
            20250000 20500000 m~z-#+xx=xx:...
        Need to +1 for end-of-last-frame
            102 --> 0.5100000
            404 --> 2.0250000
        '''
        file_id_list = read_file_list(file_id_list_file)
        file_id_list.sort(key=self.sort_by_speaker_id)

        file_id_list_num_sil_frame = self.file_id_list_num_sil_frame

        write_file_name = os.path.join(full_dir_name, 'segments')
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id in file_id_list:
                orig_file_id = unpad_speaker_id(file_id)
                n_t, n_1, n_2 = file_id_list_num_sil_frame[orig_file_id]
                t_1 = float(n_1) / 200.
                t_2 = float(n_2+1) / 200.
                f.write('%s %s %.7f %.7f\n' % (file_id, file_id, t_1, t_2))

    def make_spk2utt(self, full_dir_name, file_id_list_file):
        '''
        Write file named "spk2utt"
        spk2utt:
        p225 p225_001 p225_002 p225_003 ...
        '''
        file_id_list = read_file_list(file_id_list_file)
        file_id_list.sort(key=self.sort_by_speaker_id)

        speaker_id_list = self.dv_y_cfg.speaker_id_list_dict['all']
        spk2utt_dict = self.FLS.sort_by_speaker_list(file_id_list, speaker_id_list)

        write_file_name = os.path.join(full_dir_name, 'spk2utt')
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for speaker_id in speaker_id_list:
                spk2utt_list = spk2utt_dict[speaker_id]
                if len(spk2utt_list) > 0:
                    f.write(speaker_id)
                    for file_id in spk2utt_list:
                        f.write(' '+file_id)
                    f.write('\n')

    def make_text(self, full_dir_name, file_id_list_file):
        '''
        Write file named "text"
        text:
        p225_001 Please call Stella.
        '''
        file_id_list = read_file_list(file_id_list_file)
        file_id_list.sort(key=self.sort_by_speaker_id)

        text_dir = os.path.join(self.dv_y_cfg.raw_data_dir, 'txt')

        write_file_name = os.path.join(full_dir_name, 'text')
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id in file_id_list:
                speaker_id = file_id.split('_')[0]
                text_file_name = os.path.join(text_dir, speaker_id, '%s.txt'%file_id)
                with open(text_file_name, 'r') as f_2:
                    t = f_2.readlines()[0].strip()
                f.write('%s %s\n' % (file_id, t))

    def make_utt2spk(self, full_dir_name, file_id_list_file):
        '''
        Write file named "utt2spk"
        utt2spk:
        p225_001 p225
        '''
        file_id_list = read_file_list(file_id_list_file)
        file_id_list.sort(key=self.sort_by_speaker_id)

        write_file_name = os.path.join(full_dir_name, 'utt2spk')
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id in file_id_list:
                speaker_id = file_id.split('_')[0]
                f.write('%s %s\n' % (file_id, speaker_id))

    def make_wav_scp(self, full_dir_name, file_id_list_file):
        '''
        Write file named "wav.scp"
        "wav.scp:
        p225_001 downloads/VCTK-Corpus/wav48/p225/p225_001.wav
        '''
        file_id_list = read_file_list(file_id_list_file)
        file_id_list.sort(key=self.sort_by_speaker_id)

        write_file_name = os.path.join(full_dir_name, 'wav.scp')
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id in file_id_list:
                speaker_id = file_id.split('_')[0]
                f.write('%s downloads/VCTK-Corpus/wav24/%s/%s.wav\n' % (file_id, speaker_id, file_id))

class Make_X_Vector(object):
    '''
    Make directory named "dump/xvector"
    dump/xvector
        |-- tr_no_dev
            xvector.scp 
                p225_001 dump/xvector/tr_no_dev/xvector.1.ark:9
            spk_xvector.scp
                p225 dump/xvector/tr_no_dev/spk_xvector.ark:5
        |-- dev
        |-- eval1

    But, we have many experiments, thus add another layer
    dump/xvector
        |-- cmp
            p001.npy p002.npy
            |-- tr_no_dev
                xvector.scp spk_xvector.scp
    '''
    def __init__(self, dv_y_cfg):
        super(Make_X_Vector, self).__init__()
        self.dv_y_cfg = dv_y_cfg

        self.x_vector_dir = os.path.join(self.dv_y_cfg.data_dirs_to_link, 'dump', 'xvector')
        prepare_script_file_path(self.x_vector_dir)

    def run(self, spk_embed_name=None, spk_embed_file=None, file_embed_file=None):
        spk_embed_name  = 'cmp'
        spk_embed_file  = '/home/dawna/tts/mw545/TorchDV/dv_cmp_baseline/dvy_cmp_lr1E-04_fpu40_LRe512L_LRe512L_Lin512L_DV512S1B161T40D3440/dv_spk_dict.dat'
        spk_embed_file  = '/home/dawna/tts/mw545/TorchDV/dv_cmp_baseline/dvy_cmp_lr1E-04_fpu40_CNN512L_LRe512L_Lin512L_DV512S10T40D86/dv_spk_dict.dat'
        # file_embed_file = '/home/dawna/tts/mw545/TorchDV/dv_cmp_baseline/dvy_cmp_lr1E-04_fpu40_LRe512L_LRe512L_Lin512L_DV512S1B161T40D3440/dv_file_dict.dat'
        # spk_embed_file = '/home/dawna/tts/mw545/TorchDV/dv_wav_sincnet/dvy_wav_lr1E-04_fpu4_Sin80_LRe512L_LRe512L_Lin512L_DV512S1B161M77T160/dv_spk_dict.dat'
        # spk_embed_name = 'sincnet'

        exp_dir_name = os.path.join(self.x_vector_dir, spk_embed_name)
        if spk_embed_file is not None:
            spk_embed_values  = pickle.load(open(spk_embed_file, 'rb'))
        if file_embed_file is not None:
            file_embed_values = pickle.load(open(file_embed_file, 'rb'))

        self.save_speaker_embed_files(spk_embed_values, exp_dir_name)
        # self.make_all_scp()

    def make_all_scp(self, exp_dir_name):
        dv_y_cfg = self.dv_y_cfg

        for dir_name in ['tr_no_dev', 'dev', 'eval1']:
            full_dir_name = os.path.join(exp_dir_name, dir_name)
            prepare_script_file_path(full_dir_name)

            if dir_name == 'tr_no_dev':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'tr_no_dev.scp')
            if dir_name == 'dev':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'valid.scp')
            if dir_name == 'eval1':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'test.scp')

            self.make_file_scp(full_dir_name, file_id_list_file, spk_embed_name)
            self.make_speaker_scp(full_dir_name, file_id_list_file, spk_embed_name)

    def save_speaker_embed_files(self, spk_embed_values, exp_dir_name):
        # dump/cmp/p001.npy
        prepare_script_file_path(exp_dir_name)
        for speaker_id in spk_embed_values.keys():
            new_speaker_id = pad_speaker_id(speaker_id)
            speaker_file_name = os.path.join(exp_dir_name, "%s.npy"%new_speaker_id)
            # spk_embed_values[speaker_id].tofile(speaker_file_name)
            numpy.save(speaker_file_name, spk_embed_values[speaker_id])

    def make_file_scp(self, full_dir_name, file_id_list_file, spk_embed_name):
        # file_id dump/xvector/{spk_embed_name}/speaker_id.npy
        file_id_list = read_file_list(file_id_list_file)

        write_file_name = os.path.join(full_dir_name, 'xvector.scp')
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id in file_id_list:
                speaker_id = file_id.split('_')[0]
                f.write('%s dump/xvector/%s/%s.npy\n' % (file_id, spk_embed_name, speaker_id))

    def make_speaker_scp(self, full_dir_name, file_id_list_file, spk_embed_name):
        # speaker_id dump/xvector/{spk_embed_name}/speaker_id.npy
        file_id_list = read_file_list(file_id_list_file)

        # extract all speaker_id
        speaker_id_list = []
        for file_id in file_id_list:
            speaker_id = file_id.split('_')[0]
            if len(speaker_id_list) == 0 or speaker_id != speaker_id_list[-1]:
                    speaker_id_list.append(speaker_id)

        write_file_name = os.path.join(full_dir_name, 'spk_xvector.scp')
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for speaker_id in speaker_id_list:
                f.write('%s dump/xvector/%s/%s.npy\n' % (speaker_id, spk_embed_name, speaker_id))

    def make_speaker_random_scp(self, full_dir_name, file_id_list_file, spk_embed_name, file_embed_values, num_files):
        # file_id dump/xvector/{spk_embed_name}/random_{num_files}/speaker_id.npy
        file_id_list = read_file_list(file_id_list_file)

        exp_dir_name = os.path.join(self.x_vector_dir, spk_embed_name)
        prepare_script_file_path(exp_dir_name)

        write_file_name = os.path.join(full_dir_name, 'xvector_%i.scp' % num_files)
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id in file_id_list:
                speaker_id = file_id.split('_')[0]
                f.write('%s dump/xvector/%s/random_%i/%s.npy\n' % (file_id, spk_embed_name, num_files, speaker_id))

def temp_make_cmp_x_vector_5():
    file_embed_file = '/home/dawna/tts/mw545/TorchDV/dv_cmp_baseline/dvy_cmp_lr1E-04_fpu40_LRe512L_LRe512L_Lin512L_DV512S1B161T40D3440/dv_file_dict.dat'
    scp_file = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage/dump/xvector/cmp/eval1/xvector_5.scp'
    scp_file_5 = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/dump/spk_model_data/cmp/eval1/random_5_file_cmp.scp'
    # p024_001 dump/spk_model_data/cmp/cmp_data_dir/p024_042.cmp dump/spk_model_data/cmp/cmp_data_dir/p024_068.cmp dump/spk_model_data/cmp/cmp_data_dir/p024_044.cmp dump/spk_model_data/cmp/cmp_data_dir/p024_048.cmp dump/spk_model_data/cmp/cmp_data_dir/p024_055.cmp
    file_id_list_file = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/downloads/VCTK-Corpus/filelists/train_valid_test_SR/test.scp'
    file_id_list = read_file_list(file_id_list_file)

    spk_embed_name='cmp'
    

    # 1. write scp
    # p024_001 dump/xvector/cmp/random_5/p024.npy
    print('Writing to %s' % scp_file)
    with open(scp_file, 'w') as f:
        for file_id in file_id_list:
            # speaker_id = file_id.split('_')[0]
            f.write('%s dump/xvector/%s/random_%i/%s.npy\n' % (file_id, spk_embed_name, 5, file_id))

    # 2. make x vectors and save them
    # Note: this is a wrong version; but the model is trained with it....
    # num_frames += self.dv_file_dict[file_id][0]
    # dv_speaker += self.dv_file_dict[file_id][1]
    x_vector_dir = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage/dump/xvector/cmp/random_5'
    prepare_script_file_path(x_vector_dir)
    file_embed_values = pickle.load(open(file_embed_file, 'rb'))

    with open(scp_file_5, 'r') as f:
        scp_file_5_list = f.readlines()
    for file_id, file_line in zip(file_id_list, scp_file_5_list):
        a = file_line.strip().split(' ')
        assert file_id == a[0]

        dv_speaker = numpy.zeros(512, dtype=float)
        num_frames = 0.
        for i in range(1, len(a)):
            x_file_id = a[i].split('/')[-1].split('.')[0]
            x_file_id = unpad_speaker_id(x_file_id)

            num_frames += file_embed_values[x_file_id][0]
            dv_speaker += file_embed_values[x_file_id][1]
        dv_values = dv_speaker / num_frames

        dv_file_name = os.path.join(x_vector_dir, "%s.npy" % file_id)
        numpy.save(dv_file_name, dv_values)
    



class Make_Spk_Embed_Model_Data(object):
    '''
    "${dumpdir}/spk_model_data/${spk_model_name}/${train_set}"
    Make directory named "dump/spk_model_data"

    But, we have many experiments, thus add another layer
    dump/spk_model_data
        |-- cmp
            cmp_data_dir (norm_resil)
            |-- tr_no_dev
                same_file_cmp.scp 5_file_cmp.scp
    '''
    def __init__(self, dv_y_cfg):
        super().__init__()
        self.dv_y_cfg = dv_y_cfg

        self.spk_model_data_dir = os.path.join(self.dv_y_cfg.data_dirs_to_link, 'dump', 'spk_model_data')
        prepare_script_file_path(self.spk_model_data_dir)

        self.FLS = File_List_Selecter()

    def run(self):
        # self.run_temp()
        # self.run_make_listening_test()
        self.write_spk_embed_eval_scp()

    def run_make_listening_test(self):
        file_id_list = ['p024_001','p026_001','p141_001','p178_001','p290_001']
        spk_model_data_name = 'cmp'
        exp_dir_name = os.path.join(self.spk_model_data_dir, spk_model_data_name)

        target_dir = os.path.join(exp_dir_name, 'listening_tests')
        source_dir = os.path.join(exp_dir_name, 'eval1')
        file_name = 'same_50_seconds_per_speaker_draw_1_file_cmp.scp'

        target_file = os.path.join(target_dir, file_name)
        source_file = os.path.join(source_dir, file_name)

        useful_lines = []
        print('Reading %s' % source_file)
        with open(source_file, 'r') as f:
            l_list = f.readlines()

        for l in l_list:
            if l.split(' ')[0] in file_id_list:
                useful_lines.append(l)

        print('Writing to %s' % target_file)
        with open(target_file, 'w') as f:
            for l in useful_lines:
                f.write(l)

    def run_temp(self):
        spk_model_data_name = 'cmp'
        exp_dir_name = os.path.join(self.spk_model_data_dir, spk_model_data_name)

        if spk_model_data_name == 'cmp':
            self.cmp_data_dir = os.path.join(exp_dir_name, 'cmp_data_dir')
            # prepare_script_file_path(self.cmp_data_dir)
            # self.prepare_cmp_dir(exp_dir_name)

        # for dir_name in ['tr_no_dev', 'dev', 'eval1']:
        for dir_name in ['eval1']:
            full_dir_name = os.path.join(exp_dir_name, dir_name)

            if dir_name == 'tr_no_dev':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'tr_no_dev.scp')
                file_id_list_file_SR = os.path.join(dv_y_cfg.file_id_list_dir,'tr_no_dev.scp')
            if dir_name == 'dev':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'valid.scp')
                file_id_list_file_SR = os.path.join(dv_y_cfg.file_id_list_dir,'valid_SR.scp')
            if dir_name == 'eval1':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'test.scp')
                file_id_list_file_SR = os.path.join(dv_y_cfg.file_id_list_dir,'test_SR.scp')

            # self.write_same_file_cmp_scp(full_dir_name, file_id_list_file)
            # self.write_random_n_file_cmp_scp(full_dir_name, file_id_list_file, file_id_list_file_SR, 5)
            # self.write_all_file_cmp_scp(full_dir_name, file_id_list_file, file_id_list_file_SR)

            # for i in range(30):
            #     self.write_n_files_per_speaker_cmp_scp_multi_draws(full_dir_name, file_id_list_file, file_id_list_file_SR, n=i+1, n_draws=30)
            t_list = [5,10,15,20,30,40,50] # Shortest is 59 seconds
            for t in t_list:
                # self.write_n_seconds_per_speaker_cmp_scp(full_dir_name, file_id_list_file, file_id_list_file_SR, t=t)
                # self.write_n_seconds_per_speaker_cmp_scp_multi_draws(full_dir_name, file_id_list_file, file_id_list_file_SR, t=t, n_draws=30)
                full_dir_name = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage/dump/xvector/cmp_frame/eval_shared'
                self.write_n_seconds_per_speaker_multi_draws(full_dir_name, file_id_list_file, file_id_list_file_SR, t=t, n_draws=30)
            # self.write_dynamic_cmp_scp(full_dir_name, file_id_list_file, n=5, draw_rule='seconds')

        # self.write_spk_embed_eval_scp()
        # self.compute_all_SR_time()

    def write_dynamic_cmp_scp(self, full_dir_name, file_id_list_file, n=5, draw_rule='seconds'):
        '''
        p001_001 p001_n_type
        draw_rule: seconds; files
        '''
        prepare_script_file_path(full_dir_name)
        file_id_list = read_file_list(file_id_list_file)

        write_file_name = os.path.join(full_dir_name, 'dynamic_%i_%s_file_cmp.scp' % (n, draw_rule))
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id in file_id_list:
                speaker_id = file_id.split('_')[0]
                f.write('%s %s_%i_%s\n'% (file_id, speaker_id, n, draw_rule))

    def write_spk_embed_eval_scp(self):
        exp_dir_name = os.path.join(self.spk_model_data_dir, 'shared_directories')
        dir_name = 'spk_embed_eval1'
        full_dir_name = os.path.join(exp_dir_name, dir_name)
        prepare_script_file_path(full_dir_name)

        speaker_id_list = dv_y_cfg.speaker_id_list_dict['test']
        file_id_list_file_SR = os.path.join(dv_y_cfg.file_id_list_dir,'test_SR.scp')

        file_id_list_SR = read_file_list(file_id_list_file_SR)
        file_id_list_SR_dict = self.FLS.sort_by_speaker_list(file_id_list_SR, speaker_id_list)

        write_file_name = os.path.join(full_dir_name, 'same_file.scp')
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id in file_id_list_SR:
                l = '%s %s\n' % (file_id, file_id)
                f.write(l)

    def write_spk_embed_eval_scp_old_cmp(self):
        spk_model_data_name = 'cmp'
        exp_dir_name = os.path.join(self.spk_model_data_dir, spk_model_data_name)
        cmp_data_dir = os.path.join(exp_dir_name, 'cmp_data_dir')
        dir_name = 'spk_embed_eval1'
        full_dir_name = os.path.join(exp_dir_name, dir_name)
        prepare_script_file_path(full_dir_name)

        speaker_id_list = dv_y_cfg.speaker_id_list_dict['test']
        file_id_list_file_SR = os.path.join(dv_y_cfg.file_id_list_dir,'test_SR.scp')

        file_id_list_SR = read_file_list(file_id_list_file_SR)
        file_id_list_SR_dict = self.FLS.sort_by_speaker_list(file_id_list_SR, speaker_id_list)

        write_file_name = os.path.join(full_dir_name, 'speaker_draw_1_30_file_per_speaker_draw_30_file_cmp.scp')
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for speaker_id in speaker_id_list:
                for i in range(1,31):
                    for j in range(1,31):
                        l = '%s_%i_%i' % (speaker_id, i, j)
                        file_id_list_SR_draw = numpy.random.choice(file_id_list_SR_dict[speaker_id], i, replace=False)
                        for file_id in file_id_list_SR_draw:
                            l +=  ' dump/spk_model_data/cmp/cmp_data_dir/%s.cmp'% file_id
                        l += '\n'
                        f.write(l)
                l = '%s_all' % speaker_id
                for file_id in file_id_list_SR_dict[speaker_id]:
                    l +=  ' dump/spk_model_data/cmp/cmp_data_dir/%s.cmp'% file_id
                l += '\n'
                f.write(l)

    def prepare_cmp_dir(self, cmp_data_dir):
        orig_cmp_dir = '/data/vectra2/tts/mw545/Data/exp_dirs/data_voicebank_16kHz/nn_cmp_resil_norm_86'
        file_list_scp = os.path.join(self.dv_y_cfg.raw_data_dir, 'filelists/file_id_list_used_cfg.scp')
        file_id_list = read_file_list(file_list_scp)

        write_file_name = 'run.sh'
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id in file_id_list:
                orig_file_name = os.path.join(orig_cmp_dir, '%s.cmp' % unpad_speaker_id(file_id) )
                new_file_name  = os.path.join(self.cmp_data_dir, '%s.cmp' % file_id)
                f.write('ln -s %s %s \n' %(orig_file_name, new_file_name))

    def write_same_file_cmp_scp(self, full_dir_name, file_id_list_file):
        '''
        Write same_file_cmp.scp
        '''
        prepare_script_file_path(full_dir_name)
        file_id_list = read_file_list(file_id_list_file)
        cmp_data_dir = 'dump/spk_model_data/cmp/cmp_data_dir'

        write_file_name = os.path.join(full_dir_name, 'same_file_cmp.scp')
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id in file_id_list:
                f.write('%s dump/spk_model_data/cmp/cmp_data_dir/%s.cmp\n'% (file_id, file_id))

    def write_n_files_per_speaker_cmp_scp(self, full_dir_name, file_id_list_file, file_id_list_file_SR, n=5):
        '''
        1. Draw n files per speaker
        2. Use the same n files for all files of that speaker
        '''
        prepare_script_file_path(full_dir_name)
        file_id_list    = read_file_list(file_id_list_file)
        file_id_list_SR = read_file_list(file_id_list_file_SR)
        cmp_data_dir = 'dump/spk_model_data/cmp/cmp_data_dir'

        speaker_id_list = self.dv_y_cfg.speaker_id_list_dict['all']
        file_id_list_SR_dict = self.FLS.sort_by_speaker_list(file_id_list_SR, speaker_id_list)

        file_id_list_SR_dict_draw = {}
        for k in file_id_list_SR_dict:
            if len(file_id_list_SR_dict[k]) > n:
                file_id_list_SR_dict_draw[k] = numpy.random.choice(file_id_list_SR_dict[k], n, replace=False)

        write_file_name = os.path.join(full_dir_name, 'same_%i_file_per_speaker_file_cmp.scp' % n)
        with open(write_file_name, 'w') as f:
            for file_id_1 in file_id_list:
                speaker_id = file_id_1.split('_')[0]
                file_id_list = file_id_list_SR_dict_draw[speaker_id]
                f.write('%s'% (file_id_1))
                for file_id_2 in file_id_list:
                    f.write(' dump/spk_model_data/cmp/cmp_data_dir/%s.cmp'% (file_id_2))
                f.write('\n')

    def write_n_files_per_speaker_cmp_scp_multi_draws(self, full_dir_name, file_id_list_file, file_id_list_file_SR, n=5, n_draws=30):
        '''
        1. Draw n files per speaker
        2. Use the same n files for all files of that speaker
        '''
        prepare_script_file_path(full_dir_name)
        file_id_list    = read_file_list(file_id_list_file)
        file_id_list_SR = read_file_list(file_id_list_file_SR)
        cmp_data_dir = 'dump/spk_model_data/cmp/cmp_data_dir'

        speaker_id_list = self.dv_y_cfg.speaker_id_list_dict['all']
        file_id_list_SR_dict = self.FLS.sort_by_speaker_list(file_id_list_SR, speaker_id_list)

        for i in range(n_draws):
            file_id_list_SR_dict_draw = {}
            for k in file_id_list_SR_dict:
                if len(file_id_list_SR_dict[k]) > n:
                    file_id_list_SR_dict_draw[k] = numpy.random.choice(file_id_list_SR_dict[k], n, replace=False)

            write_file_name = os.path.join(full_dir_name, 'same_%i_file_per_speaker_draw_%i_file_cmp.scp' % (n, i+1))
            print('Writing to %s' % write_file_name)
            with open(write_file_name, 'w') as f:
                for file_id_1 in file_id_list:
                    speaker_id = file_id_1.split('_')[0]
                    f.write('%s'% (file_id_1))
                    for file_id_2 in file_id_list_SR_dict_draw[speaker_id]:
                        f.write(' dump/spk_model_data/cmp/cmp_data_dir/%s.cmp'% (file_id_2))
                    f.write('\n')

    def write_n_seconds_per_speaker_cmp_scp(self, full_dir_name, file_id_list_file, file_id_list_file_SR, t=5):
        '''
        1. For each speaker: draw files less than t seconds total (at least 1 file)
        2. Use the same n files for all files of that speaker
        '''
        prepare_script_file_path(full_dir_name)
        file_id_list    = read_file_list(file_id_list_file)
        file_id_list_SR = read_file_list(file_id_list_file_SR)
        cmp_data_dir = 'dump/spk_model_data/cmp/nn_cmp_resil_norm_86'

        speaker_id_list = self.dv_y_cfg.speaker_id_list_dict['all']
        file_id_list_SR_dict = self.FLS.sort_by_speaker_list(file_id_list_SR, speaker_id_list)

        self.DMLFIO = Data_Meta_List_File_IO()
        self.file_id_list_num_sil_frame = self.DMLFIO.read_file_list_num_silence_frame('/data/vectra2/tts/mw545/Data/exp_dirs/data_voicebank_24kHz/file_id_lists/data_meta/file_id_list_num_sil_frame.scp')

        file_id_list_SR_dict_draw = {}
        for k in file_id_list_SR_dict:
            if len(file_id_list_SR_dict[k]) > 0:
                file_id_list_SR_dict_draw[k] = self.draw_t_seconds_or_less(file_id_list_SR_dict[k], t)

        write_file_name = os.path.join(full_dir_name, 'same_%i_seconds_per_speaker_file_cmp.scp' % t)
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id_1 in file_id_list:
                speaker_id = file_id_1.split('_')[0]
                f.write('%s'% (file_id_1))
                for file_id_2 in file_id_list_SR_dict_draw[speaker_id]:
                    f.write(' %s/%s/%s.cmp'% (cmp_data_dir, speaker_id, file_id_2))
                f.write('\n')

    def write_n_seconds_per_speaker(self, full_dir_name, file_id_list_file, file_id_list_file_SR, t=5):
        '''
        1. For each speaker: draw files less than t seconds total (at least 1 file)
        2. Use the same n files for all files of that speaker
        '''
        prepare_script_file_path(full_dir_name)
        file_id_list    = read_file_list(file_id_list_file)
        file_id_list_SR = read_file_list(file_id_list_file_SR)
        cmp_data_dir = 'dump/spk_model_data/cmp/nn_cmp_resil_norm_86'

        speaker_id_list = self.dv_y_cfg.speaker_id_list_dict['all']
        file_id_list_SR_dict = self.FLS.sort_by_speaker_list(file_id_list_SR, speaker_id_list)

        self.DMLFIO = Data_Meta_List_File_IO()
        self.file_id_list_num_sil_frame = self.DMLFIO.read_file_list_num_silence_frame('/data/vectra2/tts/mw545/Data/exp_dirs/data_voicebank_24kHz/file_id_lists/data_meta/file_id_list_num_sil_frame.scp')

        file_id_list_SR_dict_draw = {}
        for k in file_id_list_SR_dict:
            if len(file_id_list_SR_dict[k]) > 0:
                file_id_list_SR_dict_draw[k] = self.draw_t_seconds_or_less(file_id_list_SR_dict[k], t)

        write_file_name = os.path.join(full_dir_name, 'same_%i_seconds_per_speaker.scp' % t)
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id_1 in file_id_list:
                speaker_id = file_id_1.split('_')[0]
                f.write('%s'% (file_id_1))
                for file_id_2 in file_id_list_SR_dict_draw[speaker_id]:
                    f.write(' %s'% (file_id_2))
                f.write('\n')

    def write_n_seconds_per_speaker_cmp_scp_multi_draws(self, full_dir_name, file_id_list_file, file_id_list_file_SR, t=5, n_draws=30):
        '''
        1. For each speaker: draw files less than t seconds total (at least 1 file)
        2. Use the same n files for all files of that speaker
        '''
        prepare_script_file_path(full_dir_name)
        file_id_list    = read_file_list(file_id_list_file)
        file_id_list_SR = read_file_list(file_id_list_file_SR)
        cmp_data_dir = 'dump/spk_model_data/cmp/nn_cmp_resil_norm_86'

        speaker_id_list = self.dv_y_cfg.speaker_id_list_dict['all']
        file_id_list_SR_dict = self.FLS.sort_by_speaker_list(file_id_list_SR, speaker_id_list)

        self.DMLFIO = Data_Meta_List_File_IO()
        self.file_id_list_num_sil_frame = self.DMLFIO.read_file_list_num_silence_frame('/data/vectra2/tts/mw545/Data/exp_dirs/data_voicebank_24kHz/file_id_lists/data_meta/file_id_list_num_sil_frame.scp')

        for i in range(n_draws):
            file_id_list_SR_dict_draw = {}
            for k in file_id_list_SR_dict:
                if len(file_id_list_SR_dict[k]) > 0:
                    file_id_list_SR_dict_draw[k] = self.draw_t_seconds_or_less(file_id_list_SR_dict[k], t)

            write_file_name = os.path.join(full_dir_name, 'same_%i_seconds_per_speaker_draw_%i_file_cmp.scp' % (t,i+1))
            print('Writing to %s' % write_file_name)
            with open(write_file_name, 'w') as f:
                for file_id_1 in file_id_list:
                    speaker_id = file_id_1.split('_')[0]
                    f.write('%s'% (file_id_1))
                    for file_id_2 in file_id_list_SR_dict_draw[speaker_id]:
                        f.write(' %s/%s/%s.cmp'% (cmp_data_dir, speaker_id, file_id_2))
                    f.write('\n')

    def write_n_seconds_per_speaker_multi_draws(self, full_dir_name, file_id_list_file, file_id_list_file_SR, t=5, n_draws=30):
        '''
        1. For each speaker: draw files less than t seconds total (at least 1 file)
        2. Use the same n files for all files of that speaker
        '''
        prepare_script_file_path(full_dir_name)
        file_id_list    = read_file_list(file_id_list_file)
        file_id_list_SR = read_file_list(file_id_list_file_SR)
        cmp_data_dir = 'dump/spk_model_data/cmp/nn_cmp_resil_norm_86'

        speaker_id_list = self.dv_y_cfg.speaker_id_list_dict['all']
        file_id_list_SR_dict = self.FLS.sort_by_speaker_list(file_id_list_SR, speaker_id_list)

        self.DMLFIO = Data_Meta_List_File_IO()
        self.file_id_list_num_sil_frame = self.DMLFIO.read_file_list_num_silence_frame('/data/vectra2/tts/mw545/Data/exp_dirs/data_voicebank_24kHz/file_id_lists/data_meta/file_id_list_num_sil_frame.scp')

        for i in range(n_draws):
            file_id_list_SR_dict_draw = {}
            for k in file_id_list_SR_dict:
                if len(file_id_list_SR_dict[k]) > 0:
                    file_id_list_SR_dict_draw[k] = self.draw_t_seconds_or_less(file_id_list_SR_dict[k], t)

            write_file_name = os.path.join(full_dir_name, 'same_%i_seconds_per_speaker_draw_%i.scp' % (t,i+1))
            print('Writing to %s' % write_file_name)
            with open(write_file_name, 'w') as f:
                for file_id_1 in file_id_list:
                    speaker_id = file_id_1.split('_')[0]
                    f.write('%s'% (file_id_1))
                    for file_id_2 in file_id_list_SR_dict_draw[speaker_id]:
                        f.write(' %s'% (file_id_2))
                    f.write('\n')

    def draw_t_seconds_or_less(self, file_id_list, t):
        t_sum = 0.
        file_id_list_draw = []
        while t_sum < t:
            file_id = numpy.random.choice(file_id_list)
            if file_id not in file_id_list_draw:
                n_t, n_1, n_2 = self.file_id_list_num_sil_frame[file_id]
                t_file = float(n_2-n_1+1) / 200.
                t_sum += t_file
                if t_sum < t or len(file_id_list_draw) == 0:
                    file_id_list_draw.append(file_id)

        return file_id_list_draw

    def compute_all_SR_time(self):
        self.DMLFIO = Data_Meta_List_File_IO()
        self.file_id_list_num_sil_frame = self.DMLFIO.read_file_list_num_silence_frame('/data/vectra2/tts/mw545/Data/exp_dirs/data_voicebank_24kHz/file_id_lists/data_meta/file_id_list_num_sil_frame.scp')

        file_id_list_file_SR = os.path.join(self.dv_y_cfg.file_id_list_dir,'test_SR.scp')
        file_id_list_SR = read_file_list(file_id_list_file_SR)

        speaker_id_list = self.dv_y_cfg.speaker_id_list_dict['all']
        file_id_list_SR_dict = self.FLS.sort_by_speaker_list(file_id_list_SR, speaker_id_list)

        for k in file_id_list_SR_dict:
            file_id_list = file_id_list_SR_dict[k]
            if len(file_id_list) > 0:
                t_sum = 0.
                for file_id in file_id_list:
                    n_t, n_1, n_2 = self.file_id_list_num_sil_frame[file_id]
                    t_file = float(n_2-n_1+1) / 200.
                    t_sum += t_file
                print('%s %f' % (k, t_sum))

    def write_random_n_file_cmp_scp(self, full_dir_name, file_id_list_file, file_id_list_file_SR, n=1):
        '''
        Write random_1_file_cmp.scp
        For each file_id in file_id_list_file, draw a random file from file_id_list_file_SR, of the same speaker_id
        '''
        prepare_script_file_path(full_dir_name)
        file_id_list    = read_file_list(file_id_list_file)
        file_id_list_SR = read_file_list(file_id_list_file_SR)
        cmp_data_dir = 'dump/spk_model_data/cmp/cmp_data_dir'

        speaker_id_list = self.dv_y_cfg.speaker_id_list_dict['all']
        file_id_list_SR_dict = self.FLS.sort_by_speaker_list(file_id_list_SR, speaker_id_list)

        write_file_name = os.path.join(full_dir_name, 'random_%i_file_cmp.scp' % n)
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id_1 in file_id_list:
                speaker_id = file_id_1.split('_')[0]
                file_id_list = numpy.random.choice(file_id_list_SR_dict[speaker_id], n, replace=False)
                f.write('%s'% (file_id_1))
                for file_id_2 in file_id_list:
                    f.write(' dump/spk_model_data/cmp/cmp_data_dir/%s.cmp'% (file_id_2))
                f.write('\n')

    def write_all_file_cmp_scp(self, full_dir_name, file_id_list_file, file_id_list_file_SR):
        '''
        Write random_1_file_cmp.scp
        For each file_id in file_id_list_file, draw a random file from file_id_list_file_SR, of the same speaker_id
        '''
        prepare_script_file_path(full_dir_name)
        file_id_list    = read_file_list(file_id_list_file)
        file_id_list_SR = read_file_list(file_id_list_file_SR)
        cmp_data_dir = 'dump/spk_model_data/cmp/cmp_data_dir'

        speaker_id_list = self.dv_y_cfg.speaker_id_list_dict['all']
        file_id_list_SR_dict = self.FLS.sort_by_speaker_list(file_id_list_SR, speaker_id_list)

        write_file_name = os.path.join(full_dir_name, 'all_file_cmp.scp')
        print('Writing to %s' % write_file_name)
        with open(write_file_name, 'w') as f:
            for file_id_1 in file_id_list:
                speaker_id = file_id_1.split('_')[0]
                file_id_list = file_id_list_SR_dict[speaker_id]
                f.write('%s'% (file_id_1))
                for file_id_2 in file_id_list:
                    f.write(' dump/spk_model_data/cmp/cmp_data_dir/%s.cmp'% (file_id_2))
                f.write('\n')


def setup_2_stage_exp_directory(dv_y_cfg):
    setup_script = 'setup.sh'
    dataset_name = 'CUED_vctk'
    exp_name = 'tts_2_stage'

    write_file_name = setup_script
    print('Writing to %s' % write_file_name)
    with open(write_file_name, 'w') as f:
        f.write('cd %s\n' % dv_y_cfg.espnet_root)
        f.write('egs2/TEMPLATE/tts1/setup.sh egs2/%s/%s\n' % (dataset_name, exp_name))
        f.write('cd egs2/%s/%s\n' % (dataset_name, exp_name))
        f.write('cp ../../mini_an4/tts1/run.sh .\n')
        f.write('ln -s %s/* .\n' % dv_y_cfg.data_dirs_to_link)
        f.write('rm tts.sh\n')
        f.write('ln -s ../../../espnet2_modified_CUED/tts.sh\n')
        f.write('mkdir log\n')

        # Scripts in previous vctk exp dir
        f.write('cp ../../vctk/tts_xvector/run.sh .\n')
        f.write('cp ../../vctk/tts_xvector/run_grid.sh .\n')
        f.write('cp ../../vctk/tts_xvector/submit_grid.sh .\n')
        f.write('cp ../../vctk/tts_xvector/conf/* conf/ -r \n')


    print('Run setup.sh')
    print('Change run_grid.sh')
    print('Run step 2 in run_grid.sh')

def setup_integrated_exp_directory(dv_y_cfg):

    setup_script = 'setup.sh'
    dataset_name = 'CUED_vctk'
    exp_name = 'tts_integrated'

    write_file_name = setup_script
    print('Writing to %s' % write_file_name)
    with open(write_file_name, 'w') as f:
        f.write('cd %s\n' % dv_y_cfg.espnet_root)
        f.write('egs2/TEMPLATE/tts1/setup.sh egs2/%s/%s\n' % (dataset_name, exp_name))
        f.write('cd egs2/%s/%s\n' % (dataset_name, exp_name))
        f.write('cp ../../mini_an4/tts1/run.sh .\n')
        f.write('ln -s %s/* .\n' % dv_y_cfg.data_dirs_to_link)
        f.write('rm tts.sh\n')
        f.write('ln -s ../../../espnet2_modified_CUED/tts.sh\n')
        f.write('mkdir log\n')

        # Scripts in previous vctk exp dir
        f.write('cp ../../vctk/tts_gst/run.sh .\n')
        f.write('cp ../../vctk/tts_gst/run_grid.sh .\n')
        f.write('cp ../../vctk/tts_gst/submit_grid.sh .\n')
        f.write('cp ../../vctk/tts_gst/conf/* conf/ -r \n')


    print('Run setup.sh')
    print('Change run_grid.sh')
    print('Run step 2 in run_grid.sh')

def temp_change_1():
    # Temp script: after move kaldi-style dump/xvector to dump/xvector/xvector, modify .scp files
    # Run once in each directory, tr_no_dev, dev, eval1
    import os
    file_list = os.listdir('.')
    for file_name in file_list:
        if file_name.split('.')[-1] == 'scp':
            with open(file_name, 'r') as f_1:
                line_list_1 = f_1.readlines()
            line_list_2 = [x.replace('dump/xvector', 'dump/xvector/xvector') for x in line_list_1]
            with open(file_name, 'w') as f_2:
                f_2.writelines(line_list_2)

def remove_norm_denorm_from_exp_directory():
    # Remove norm and denorm folders to free disk space
    func_logger = make_logger('remove_norm_denorm')
    # exp_dir_name_list = ['tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_5_seconds_copy_%i' % i for i in range(1,5)]
    exp_dir_name_list = ['tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_15_seconds']
    for exp_dir_name in exp_dir_name_list:
        start_dir = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/exp/%s/decode_valid.loss.best' % exp_dir_name
        func_logger.info('remove from %s' % start_dir)
        folder_list = os.listdir(start_dir)
        for folder_name in folder_list:
            folder_name_split = folder_name.split('_')
            if len(folder_name_split) > 2 and folder_name_split[0] == 'same':
                start_dir_2 = os.path.join(start_dir, folder_name, 'eval1/log')
                folder_list_2 = os.listdir(start_dir_2)
                for folder_name_2 in folder_list_2:
                    folder_name_split_2 = folder_name_2.split('.')
                    if len(folder_name_split_2) > 1 and folder_name_split_2[0] == 'output':
                        start_dir_3 = os.path.join(start_dir_2, folder_name_2)
                        folder_list_3 = os.listdir(start_dir_3)
                        if 'norm' in folder_list_3:
                            norm_dir = os.path.join(start_dir_3, 'norm')
                            print('removing %s' % norm_dir, flush=True)
                            shutil.rmtree(norm_dir)
                        if 'denorm' in folder_list_3:
                            denorm_dir = os.path.join(start_dir_3, 'denorm')
                            print('removing %s' % denorm_dir, flush=True)
                            shutil.rmtree(denorm_dir)

class Make_dir_listening_test(object):
    """
    This class makes a directory
    sample_dir
      |-- title_1
        |-- file_id_1.wav file_id_2.wav
      |-- title_2
    The wav files are full-path links
    """
    def __init__(self):
        super(Make_dir_listening_test, self).__init__()

        self.file_dir_list = {}
        self.file_id_list = ['p024_001','p026_001','p141_001','p178_001','p290_001'] + ['p024_004','p026_004','p141_004','p178_004','p290_004']
        self.sample_tool_dir = '/home/dawna/tts/mw545/tools/weblisteningtest'
        self.sample_root_dir = os.path.join(self.sample_tool_dir, 'samples-mw545')

    def run(self):
        self.file_dir_type = {}
        self.file_dir_type['orig'] = ['/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/data_speaker_206/downloads/VCTK-Corpus/wav24','speaker_id/file_id.wav']

        
        # self.setting_8()
        # self.make_demo_dirs()
        # self.write_demo_command()

        self.setting_7()
        self.make_cmos_dirs()
        # self.write_cmos_command()

    def setting_1(self):
        self.table_name = 'Voicebank_ESPNet2_Tacotron2_vocoder_2_stage_cmp_spk_model_amounts'
        self.title_list = ['orig','cmp_spk_model_5s', 'cmp_spk_model_frame']
        work_dir_2_stage = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage'

        self.file_dir_type['cmp_spk_model_5s'] = [os.path.join(work_dir_2_stage, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_5s_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval_shared/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['cmp_spk_model_frame'] = [os.path.join(work_dir_2_stage, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_frame_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval_shared/wav_pwg'), 'file_id_gen.wav']

    def setting_2(self):
        self.table_name = 'Voicebank_ESPNet2_Tacotron2_vocoder_2_stage_add_vs_concat'
        self.title_list = ['orig','cmp_train_5s_add', 'cmp_train_5s_concat','cmp_train_30s_add','cmp_train_30s_concat']
        work_dir_2_stage = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage'

        self.file_dir_type['cmp_train_5s_add'] = [os.path.join(work_dir_2_stage, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_5s_dynamic_5_seconds/decode_valid.loss.best/same_5_seconds_per_speaker_draw_1/eval_shared/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['cmp_train_5s_concat'] = [os.path.join(work_dir_2_stage, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_5s_concat_dynamic_5_seconds/decode_valid.loss.best/same_5_seconds_per_speaker_draw_1/eval_shared/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['cmp_train_30s_add'] = [os.path.join(work_dir_2_stage, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_5s_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval_shared/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['cmp_train_30s_concat'] = [os.path.join(work_dir_2_stage, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_5s_concat_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval_shared/wav_pwg'), 'file_id_gen.wav']

    def setting_3(self):
        self.table_name = 'Voicebank_ESPNet2_Tacotron2_vocoder_2_stage_cmp_vs_sincnet'
        self.title_list = ['orig','cmp_spk_model_frame', 'sincnet_spk_model_frame']
        work_dir_2_stage = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage'

        self.file_dir_type['cmp_spk_model_frame'] = [os.path.join(work_dir_2_stage, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_frame_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval_shared/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['sincnet_spk_model_frame'] = [os.path.join(work_dir_2_stage, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_sincnet_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval_shared/wav_pwg'), 'file_id_gen.wav']

    def setting_4(self):
        self.table_name = 'Voicebank_ESPNet2_Tacotron2_vocoder_cmp_2_stage_vs_integrated'
        self.title_list = ['orig','cmp_2_stage', 'cmp_integrated']
        work_dir_2_stage = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage'
        work_dir_integrated = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated'

        self.file_dir_type['cmp_2_stage'] = [os.path.join(work_dir_2_stage, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_5s_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval_shared/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['cmp_integrated'] = [os.path.join(work_dir_integrated, 'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_5_seconds_add_backup/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']

    def setting_5(self):
        self.table_name = 'Interspeech_48'
        # self.title_list = ['orig','cmp', 'sincnet', 'sinenet']
        self.title_list = ['orig','cmp', 'sincnet_4800', 'sinenet_4800']
        work_dir_integrated = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated'

        self.file_dir_type['cmp'] = [os.path.join(work_dir_integrated, 'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['sincnet'] = [os.path.join(work_dir_integrated, 'exp/tts_train_sincnet_tacotron2_raw_phn_tacotron_g2p_en_no_space_sincnet_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['sincnet_4800'] = [os.path.join(work_dir_integrated, 'exp/tts_train_sincnet_4800_tacotron2_raw_phn_tacotron_g2p_en_no_space_sincnet_4800_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['sinenet'] = [os.path.join(work_dir_integrated, 'exp/tts_train_sinenet_tacotron2_raw_phn_tacotron_g2p_en_no_space_sinenet_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['sinenet_4800'] = [os.path.join(work_dir_integrated, 'exp/tts_train_sinenet_4800_tacotron2_raw_phn_tacotron_g2p_en_no_space_sinenet_4800_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']

    def setting_6(self):
        self.table_name = 'Interspeech_2I'
        self.title_list = ['orig','cmp', 'sincnet', 'sinenet']
        # self.title_list = ['orig','sinenet_2', 'sinenet_I']
        work_dir_2_stage = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage'
        work_dir_integrated = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated'

        self.file_dir_type['sinenet_2'] = [os.path.join(work_dir_2_stage, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_sinenet_frame_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['sinenet_I'] = [os.path.join(work_dir_integrated, 'exp/tts_train_sinenet_tacotron2_raw_phn_tacotron_g2p_en_no_space_sinenet_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']

    def setting_7(self):
        self.table_name = 'Interspeech_I'
        self.title_list = ['orig','cmp', 'sincnet', 'sinenet']
        # self.title_list = ['orig','sinenet_2', 'sinenet_I']
        work_dir_2_stage = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage'
        work_dir_integrated = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated'
        self.file_dir_type['cmp'] = [os.path.join(work_dir_integrated, 'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['sincnet'] = [os.path.join(work_dir_integrated, 'exp/tts_train_sincnet_tacotron2_raw_phn_tacotron_g2p_en_no_space_sincnet_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['sinenet'] = [os.path.join(work_dir_integrated, 'exp/tts_train_sinenet_tacotron2_raw_phn_tacotron_g2p_en_no_space_sinenet_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']

    def setting_8(self):
        self.table_name = 'Voicebank_ESPNet2_Tacotron2_vocoder_sincnet_sinenet_4800'
        self.title_list = ['orig','cmp', 'sincnet', 'sincnet_4800', 'sinenet', 'sinenet_4800']
        work_dir_integrated = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated'

        self.file_dir_type['cmp'] = [os.path.join(work_dir_integrated, 'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['sincnet'] = [os.path.join(work_dir_integrated, 'exp/tts_train_sincnet_tacotron2_raw_phn_tacotron_g2p_en_no_space_sincnet_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['sincnet_4800'] = [os.path.join(work_dir_integrated, 'exp/tts_train_sincnet_4800_tacotron2_raw_phn_tacotron_g2p_en_no_space_sincnet_4800_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['sinenet'] = [os.path.join(work_dir_integrated, 'exp/tts_train_sinenet_tacotron2_raw_phn_tacotron_g2p_en_no_space_sinenet_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']
        self.file_dir_type['sinenet_4800'] = [os.path.join(work_dir_integrated, 'exp/tts_train_sinenet_4800_tacotron2_raw_phn_tacotron_g2p_en_no_space_sinenet_4800_dynamic_5_seconds/decode_valid.loss.best/same_30_seconds_per_speaker_draw_1/eval1/wav_pwg'), 'file_id_gen.wav']


    def make_date_str(self):
        '''
        yyyy_mm_dd format
        '''
        from datetime import datetime
        return datetime.today().strftime('%Y_%m_%d')


    def make_file_name(self, file_id, file_dir_type):
        dir_name  = file_dir_type[0]
        file_type = file_dir_type[1]
        new_file_name = file_type.replace('file_id',file_id)
        if 'speaker_id' in file_type:
            s = file_id.split('_')[0]
            new_file_name = new_file_name.replace('speaker_id',s)

        new_file_name_full = os.path.join(dir_name, new_file_name)
        return new_file_name_full

    def make_demo_dirs(self):
        date_str = self.make_date_str()
        self.sample_dir = os.path.join(self.sample_root_dir, date_str+ '_' + self.table_name)
        prepare_script_file_path(self.sample_dir)
        run_script = os.path.join(self.sample_dir, 'run_1.sh')
        
        with open(run_script, 'w') as f:
            for title_name in self.title_list:
                title_dir = os.path.join(self.sample_dir, title_name)
                prepare_script_file_path(title_dir)

                for file_id in self.file_id_list:
                    orig_file_name = self.make_file_name(file_id, self.file_dir_type[title_name])
                    new_file_name  = os.path.join(title_dir, file_id+'.wav')
                    f.write('ln -s %s %s\n' %(orig_file_name, new_file_name))

        print('bash %s \n' % run_script)

    def write_demo_command(self):
        '''e.g.
        python demotable_html_wav.py --idexp "^(p[0-9]+_[0-9]+)"  --tablename VoiceBank_Tacotron2 \
        /home/dawna/tts/mw545/Export_Temp/output_inference/orig /home/dawna/tts/mw545/Export_Temp/output_inference/mel_true /home/dawna/tts/mw545/Export_Temp/output_inference/mel_gen  \
        --titles original  mel-true mel-gen-tacotron'
        '''

        # run_script = os.path.join(self.sample_dir, 'run_2.sh')
        run_script = 'run_2.sh'
        with open(run_script, 'w') as f:
            f.write('\nCurrent working dir is %s\n' % os.path.dirname(os.path.realpath(__file__)))
            f.write('\n'*2)
            f.write('cd %s \n' % self.sample_tool_dir)

            f.write('python demotable_html_wav.py --idexp "^(p[0-9]+_[0-9]+)"  --tablename %s  --outhtml %s.html \\ \n' % (self.table_name, self.table_name))
            dir_list = [self.sample_dir + '/' + t for t in self.title_list]
            f.write(' '.join(dir_list) + ' \\ \n')
            f.write('--titles %s \n' %(' '.join(self.title_list)))

            date_str = self.make_date_str()
            weblink_dir = os.path.join(self.sample_tool_dir, 'Samples_Webpage_Dir', date_str+ '_' + self.table_name)
            f.write('rm -r %s \n' % weblink_dir)
            f.write('mkdir %s \n' % weblink_dir)
            f.write('mv %s %s.html %s/ \n' % (self.table_name, self.table_name, weblink_dir))
            f.write('\n'*2)
            f.write('Add this line to %s/Samples_Webpage_Dir/2021_07_17_Voicebank_ESPNet2_Tacotron2.html: \n' % self.sample_tool_dir)
            # f.write('<li><a href="samples-mw545/%s/%s.html">%s</a> </li>\n' % (date_str+ '_' + self.table_name, self.table_name, self.table_name))
            f.write('<li><a href="%s/%s.html">%s</a> </li>\n' % (date_str+ '_' + self.table_name, self.table_name, self.table_name))

            f.write('\nGo back to %s ? \n\n' % os.path.dirname(os.path.realpath(__file__)))

        print('cat %s \n' % run_script)

    def make_cmos_dirs(self):
        date_str = self.make_date_str()
        self.sample_dir = os.path.join(self.sample_root_dir, date_str+ '_' + self.table_name)
        prepare_script_file_path(self.sample_dir)
        run_script = os.path.join(self.sample_dir, 'run_1.sh')
        file_list_scp = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/filelists/train_valid_test_SR/test.scp'
        file_id_list = read_file_list(file_list_scp)

        
        with open(run_script, 'w') as f:
            for title_name in self.title_list:
                title_dir = os.path.join(self.sample_dir, title_name)
                prepare_script_file_path(title_dir)

                for file_id in file_id_list:
                    orig_file_name = self.make_file_name(file_id, self.file_dir_type[title_name])
                    new_file_name  = os.path.join(title_dir, file_id+'.wav')
                    f.write('ln -s %s %s\n' %(orig_file_name, new_file_name))

        print('bash %s \n' % run_script)

    def write_cmos_command(self):
        '''e.g.
        python demotable_html_wav.py --idexp "^(p[0-9]+_[0-9]+)"  --tablename VoiceBank_Tacotron2 \
        /home/dawna/tts/mw545/Export_Temp/output_inference/orig /home/dawna/tts/mw545/Export_Temp/output_inference/mel_true /home/dawna/tts/mw545/Export_Temp/output_inference/mel_gen  \
        --titles original  mel-true mel-gen-tacotron'
        '''

        # run_script = os.path.join(self.sample_dir, 'run_2.sh')
        date_str = self.make_date_str()
        self.output_dir = date_str+ '_' + self.table_name
        run_script = 'run_2.sh'
        with open(run_script, 'w') as f:
            f.write('\nCurrent working dir is %s\n' % os.path.dirname(os.path.realpath(__file__)))
            f.write('\n'*2)
            f.write('cd %s \n' % self.sample_tool_dir)

            # python listtesttable_cmos.py samples/sinenet_03_25/cmp samples/sinenet_03_25/sincnet samples/sinenet_03_25/sinenet -o sinenet_03_25 --idexp "^(p[0-9]+_[0-9]+)" --tablename sinenet_03_25 --srvpath https://alta-crowdsourcing.org.uk/~rcv25/TTS-mturk-qualeval/mw545
            f.write('python listtesttable_cmos_mw545.py --idexp "^(p[0-9]+_[0-9]+)"  --tablename %s  --outdir %s --srvpath https://alta-crowdsourcing.org.uk/~rcv25/TTS-mturk-qualeval/mw545 --includeref True --nbsample 6 \\ \n' % (self.table_name, self.output_dir))
            # Move orig to the last
            self.title_list.remove('orig')
            self.title_list.append('orig')
            dir_list = [self.sample_dir + '/' + t for t in self.title_list]
            f.write(' '.join(dir_list) + ' \n')

            if False:
                date_str = self.make_date_str()
                weblink_dir = os.path.join(self.sample_tool_dir, 'Samples_Webpage_Dir', date_str+ '_' + self.table_name)
                f.write('rm -r %s \n' % weblink_dir)
                f.write('mkdir %s \n' % weblink_dir)
                f.write('cp %s  %s/ -r \n' % (self.output_dir, weblink_dir))
                f.write('\n'*2)
                f.write('Add this line to %s/Samples_Webpage_Dir/2021_07_17_Voicebank_ESPNet2_Tacotron2.html: \n' % self.sample_tool_dir)
                # f.write('<li><a href="samples-mw545/%s/%s.html">%s</a> </li>\n' % (date_str+ '_' + self.table_name, self.table_name, self.table_name))
                f.write('<li><a href="%s/%s.php">%s</a> </li>\n' % (date_str+ '_' + self.table_name, self.table_name, self.table_name))

            f.write('\nGo back to %s ? \n\n' % os.path.dirname(os.path.realpath(__file__)))

        print('cat %s \n' % run_script)



class Make_loss_plot(object):
    """
    This class makes a directory
    sample_dir
      |-- title_1
        |-- file_id_1.wav file_id_2.wav
      |-- title_2
    The wav files are full-path links
    """
    def __init__(self):
        super(Make_loss_plot, self).__init__()
        self.logger = make_logger('Make_loss_plot')

        self.loss_name_list = ['l1_loss', 'mse_loss', 'bce_loss', 'attn_loss', 'loss']
        self.graph_plotter = Graph_Plotting()

    def run(self):
        self.print_mse_30_draws()
        # self.std_plot_from_file()


    def print_mse_30_draws(self):
        dir_list = []
        work_dir = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated'
        
        
        dir_list.append('tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_5_seconds')
        dir_list.append('tts_train_sincnet_tacotron2_raw_phn_tacotron_g2p_en_no_space_sincnet_dynamic_5_seconds')
        dir_list.append('tts_train_sinenet_tacotron2_raw_phn_tacotron_g2p_en_no_space_sinenet_dynamic_5_seconds')
        # dir_list.append('tts_train_sincnet_4800_tacotron2_raw_phn_tacotron_g2p_en_no_space_sincnet_4800_dynamic_5_seconds')
        # dir_list.append('tts_train_sinenet_4800_tacotron2_raw_phn_tacotron_g2p_en_no_space_sinenet_4800_dynamic_5_seconds')

        # work_dir = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage'
        # dir_list.append('tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_frame_concat_dynamic_5_seconds')
        # dir_list.append('tts_train_raw_phn_tacotron_g2p_en_no_space_seed1_sincnet_concat_dynamic_5_seconds')
        # dir_list.append('tts_train_raw_phn_tacotron_g2p_en_no_space_sinenet_frame_dynamic_5_seconds')

        full_dir_list = []
        for dir_name in dir_list:
            full_dir_name = os.path.join(work_dir, 'exp', dir_name, 'tf/same_[n1]_seconds_per_speaker_draw_[n2]/eval1/log')
            full_dir_list.append(full_dir_name)
            # full_dir_name = os.path.join(work_dir, 'exp', dir_name, 'tf_50/same_[n1]_seconds_per_speaker_draw_[n2]/eval1/log')
            # full_dir_list.append(full_dir_name)
            
        num_sec_list = [30]
        # num_sec_list = [5,10,15,20,30,40,50]
        for log_dir_temp in full_dir_list:
            mse_mean_list = []
            mse_std_list  = []
            for i in num_sec_list:
                mse_list_i_file = []
                for j in range(1,31):
                    log_dir = log_dir_temp.replace('[n1]', str(i)).replace('[n2]', str(j))
                    total_loss_dict = self.get_losses_from_dir(log_dir, upper_limit=10)
                    mse_list_i_file.append(numpy.mean(total_loss_dict['l1_loss']))
                    # mse_list_i_file.append(numpy.mean(total_loss_dict['mse_loss']))
                mse_mean_list.append(numpy.mean(mse_list_i_file))
                # mse_std_list.append(numpy.mean(numpy.std(mse_list_i_file,axis=0,ddof=1)))
                mse_std_list.append(numpy.std(mse_list_i_file,ddof=1))

            self.logger.info('Results from')
            print('%s' % log_dir_temp)
            print(num_sec_list)
            print(mse_mean_list)
            print(mse_std_list)

    def plot_mse_5(self):
        # std is computed across different models, instead of different draws
        dir_list = []
        test_name = 'integrated-multi-models'
        work_dir = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/'
        dir_list.append(work_dir+'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_5_seconds_add_backup/tf/same_[n1]_seconds_per_speaker_draw_[n2]/eval1/log')
        dir_list.append(work_dir+'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_5_seconds_copy_1/tf/same_[n1]_seconds_per_speaker_draw_[n2]/eval1/log')
        dir_list.append(work_dir+'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_5_seconds_copy_2/tf/same_[n1]_seconds_per_speaker_draw_[n2]/eval1/log')
        dir_list.append(work_dir+'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_5_seconds_copy_3/tf/same_[n1]_seconds_per_speaker_draw_[n2]/eval1/log')
        dir_list.append(work_dir+'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_5_seconds_copy_4/tf/same_[n1]_seconds_per_speaker_draw_[n2]/eval1/log')

        mse_mean_list = []
        mse_std_list  = []
        for i in [5,10,15,20,30,40,50]:
            mse_list_files = []
            for log_dir_temp in dir_list:
                mse_list_file = []
                for j in range(1,31):
                    log_dir = log_dir_temp.replace('[n1]', str(i)).replace('[n2]', str(j))
                    total_loss_dict = self.get_losses_from_dir(log_dir)
                    mse_list_file.extend(total_loss_dict['mse_loss'])
                mse_list_files.append(mse_list_file)
            mse_mean_list.append(numpy.mean(mse_list_files))
            mse_std_list.append(numpy.mean(numpy.std(mse_list_files,axis=0,ddof=1)))

        print('Results of test: %s' % test_name)
        print(mse_mean_list)
        print(mse_std_list)

    def format_loss_dict(self, loss_dict):
        # format loss dict for better printing and fill in form
        # 4 d.p., separated by '&'
        # except attention_loss, very small, use 4 s.f.
        s = ''
        for k in self.loss_name_list:
            if k == 'attn_loss':
                s += '%.4E & ' % loss_dict[k]
            else:
                s += '%.4f & ' % loss_dict[k]
        s = s[:-2]
        s += '\\\\'
        return s

    def read_log_file(self, full_file_name):
        # return a dict of lists of loss, and number of files
        with open(full_file_name, 'r') as f:
            file_lines = f.readlines()
        file_loss_dict = {k:[] for k in self.loss_name_list}
        num_files = 0.
        for line in file_lines:
            # {'l1_loss': 0.5132738947868347, 'mse_loss': 0.22509634494781494, 'bce_loss': 0.010223600082099438, 'attn_loss': 3.832143556792289e-05, 'loss': 0.7486321926116943}
            if line[0] == '{':  # dict of losses, start with {
                num_files += 1
                k_v_pairs = line.strip().split(', ')
                for k_v_p in k_v_pairs:
                    k = k_v_p.split(':')[0].split("'")[1]
                    v = k_v_p.split(' ')[1]
                    if v[-1] == '}':
                        v = v[:-1]
                    v = float(v)
                    file_loss_dict[k].append(v)

        return file_loss_dict, num_files

    def get_losses_from_dir(self, dir_name, upper_limit=100):
        # extract loss from all log files tts_inference.*.log
        total_loss_dict = {k:[] for k in self.loss_name_list}
        total_num_file  = 0.

        file_list = os.listdir(dir_name)
        for file_name in file_list:
            if file_name.split('.')[0] == 'tts_inference' and file_name.split('.')[-1] == 'log':
                if int(file_name.split('.')[1]) <= upper_limit:
                    full_file_name = os.path.join(dir_name, file_name)
                    file_loss_dict, num_files = self.read_log_file(full_file_name)
                    for k in self.loss_name_list:
                        total_loss_dict[k].extend(file_loss_dict[k])
                        total_num_file += num_files

        if total_num_file == 0:
            print('0 files found in %s!' % dir_name)
        # return {k:total_loss_dict[k]/total_num_file for k in self.loss_name_list}
        return total_loss_dict

    def get_loss_dict(self, exp_dir_name):
        loss_list_dict = {k:[] for k in self.loss_name_list}
        for i in range(30):
            dir_name = os.path.join(exp_dir_name, 'tf_%i/eval1/log' % (i+1))
            loss_dict = self.get_losses_from_dir(dir_name)

            for k in self.loss_name_list:
                loss_list_dict[k].append(loss_dict[k])
        return loss_list_dict

    def plot_diff_num(self):
        # There are 2 models to compare all losses in self.loss_name_list
        # plot 2 curves for each loss
        exp_name_list = []
        exp_dir_name_list = []

        exp_name_list.append('Train 1')
        exp_dir_name_list.append('/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_1')
        exp_name_list.append('Train 5')
        exp_dir_name_list.append('/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_5')

        loss_dict_list = []
        for exp_dir_name in exp_dir_name_list:
            loss_dict_list.append(self.get_loss_dict(exp_dir_name))
        
        x = range(1,31)
        save_dir = '/home/dawna/tts/mw545/Export_Temp'
        # save_dir = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_5/tf'
        # plot individual losses first
        for k in self.loss_name_list:
            fig_file_name = os.path.join(save_dir, '%s.png' % k)
            x_list = [x] * len(exp_name_list)
            y_list = [loss_dict[k] for loss_dict in loss_dict_list]
            legend_list = exp_name_list
            self.graph_plotter.single_plot(fig_file_name, x_list, y_list, legend_list, title=k, x_label='num_utter_adaptation_inference', y_label=k)

        if False:
            # Do not plot all_losses; very different scales, look like flat lines
            fig_file_name = os.path.join(save_dir, 'all_losses.png')
            x_list = []
            y_list = []
            legend_list = []
            for k in self.loss_name_list:
                x_list.append(x)
                y_list.append(loss_list_dict[k])
                legend_list.append(k)
            self.graph_plotter.single_plot(fig_file_name, x_list, y_list, legend_list, title=None, x_label='num_utter_adaptation', y_label=None)

    def plot_num_seconds(self):
        '''
        Plot losses against amount of data used
        '''
        x = [5,10,15,20,25,30,35,40,45,50,55]
        loss_list_dict = {k:[] for k in self.loss_name_list}
        for i in x:
            dir_name = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_5/decode_valid.loss.best/same_%i_seconds_per_speaker/eval1/log' % i
            loss_dict = self.get_losses_from_dir(dir_name)

            for k in self.loss_name_list:
                loss_list_dict[k].append(loss_dict[k])


        save_dir = '/home/dawna/tts/mw545/Export_Temp'
        for k in self.loss_name_list:
            fig_file_name = os.path.join(save_dir, 'sec_%s.png' % k)
            x_list = [x]
            y_list = [loss_list_dict[k]]
            legend_list=[None]
            self.graph_plotter.single_plot(fig_file_name, x_list, y_list, legend_list, title=k, x_label='num_seconds_adaptation_inference', y_label=k)

    def plot_num_utterances(self):
        '''
        Plot losses against amount of data used
        '''
        x = range(1,31)
        loss_list_dict = {k:[] for k in self.loss_name_list}
        for i in x:
            dir_name = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_5/decode_valid.loss.best/same_%i_file_per_speaker/eval1/log' % i
            loss_dict = self.get_losses_from_dir(dir_name)

            for k in self.loss_name_list:
                loss_list_dict[k].append(loss_dict[k])


        save_dir = '/home/dawna/tts/mw545/Export_Temp'
        for k in self.loss_name_list:
            fig_file_name = os.path.join(save_dir, 'utter_%s.png' % k)
            x_list = [x]
            y_list = [loss_list_dict[k]]
            legend_list=[None]
            self.graph_plotter.single_plot(fig_file_name, x_list, y_list, legend_list, title=k, x_label='num_utterances_adaptation_inference', y_label=k)

    def std_plot_from_file(self, file_name='/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/scripts_CUED/input_std_plot.txt'):
        '''
        Input file format:
        Fig name; string
        value_name; string
        x; list of integers, split by ', ', e.g. [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        y_1
        s_1
        l_1
        y_2
        s_2
        l_2
        ...
        '''
        with open(file_name,'r') as f:
            file_lines = f.readlines()

        y_list = []
        s_list = []
        l_list = []

        fig_file_name = file_lines[0].strip()
        value_name = file_lines[1].strip()
        x = self.list_string_2_list(file_lines[2])
        num_plots = int((len(file_lines)-3)/3)
        
        for i in range(num_plots):
            y = self.list_string_2_list(file_lines[i*3+3])
            s = self.list_string_2_list(file_lines[i*3+4])
            l = file_lines[i*3+5].strip()
            y_list.append(y)
            s_list.append(s)
            l_list.append(l)

        self.std_plot_function(y_list, s_list, l_list, fig_file_name, x, value_name)
        # self.mean_std_split_plot_function(y_list, s_list, l_list, fig_file_name, x, value_name)

    def list_string_2_list(self, input_str):
        '''
        e.g. input:
        '[0.0005148354028148832, 0.00041915299483325863, 0.00026150449597130533, 0.00023824020019932277, 0.0001858520390259897, 0.00012209781622027536, 0.00010016345589939115]'
        output:
        [0.0005148354028148832, 0.00041915299483325863, 0.00026150449597130533, 0.00023824020019932277, 0.0001858520390259897, 0.00012209781622027536, 0.00010016345589939115]
        '''
        a = input_str.strip()[1:-1].split(', ')
        b = [float(x) for x in a]
        return b

    def std_plot_function(self, y_list, s_list, l_list, fig_file_name,x=None, value_name='mse'):
        if x is None:
            x = [5, 10, 15, 20, 30, 40, 50]
        x_list = [x] * len(y_list)

        graph_plotter = Graph_Plotting()
        save_dir = '/home/dawna/tts/mw545/Export_Temp'
        fig_file_name = os.path.join(save_dir, fig_file_name)
        title = 'mean and std of %s' % value_name
        x_label = 'max_num_seconds_adaptation_inference'
        y_label = value_name
        graph_plotter.mean_std_plot(fig_file_name, x_list, y_list, s_list, l_list, title=title, x_label=x_label, y_label=y_label)

    def mean_std_split_plot_function(self, y_list, s_list, l_list, fig_file_name,x=None, value_name='mse'):
        '''
        2 subplots, plot mean and std separately
        '''
        if x is None:
            x = [5,10,15,20,30,40,50]
        x_list = [x] * len(y_list)

        graph_plotter = Graph_Plotting()
        save_dir = '/home/dawna/tts/mw545/Export_Temp'
        fig_file_name = os.path.join(save_dir, fig_file_name)
        x_label = 'max_num_seconds_adaptation_inference'
        '''
        Plot mean
        '''
        y_label = 'mean of %s' % value_name
        title = 'mean of %s' % value_name
        new_fig_file_name = fig_file_name.replace('.png','.mean.png')
        graph_plotter.single_plot(new_fig_file_name, x_list, y_list, l_list, title, x_label, y_label)
        '''
        Plot std
        '''
        y_label = 'std of %s' % value_name
        title = 'std of %s' % value_name
        new_fig_file_name = fig_file_name.replace('.png','.std.png')
        graph_plotter.single_plot(new_fig_file_name, x_list, s_list, l_list, title, x_label, y_label)
        




class ESPNet_log_reader_plotter(object):

    def __init__(self):
        '''
        e.g. line:
        [air209] 2021-10-18 23:44:50,895 (trainer:250) INFO: 1epoch results: [train] iter_time=7.550, forward_time=0.952, l1_loss=0.989, mse_loss=0.840, bce_loss=0.130, attn_loss=0.004, loss=1.962, backward_time=0.883, optim_step_time=0.011, lr_0=0.001, train_time=9.588, time=1 hour, 19 minutes and 54.41 seconds, total_count=500, [valid] l1_loss=0.788, mse_loss=0.512, bce_loss=0.073, attn_loss=0.002, loss=1.375, time=1 minute and 56.46 seconds, total_count=13, [att_plot] time=4.54 seconds, total_count=0
        '''
        super(ESPNet_log_reader_plotter, self).__init__()

        self.loss_name_list = ['l1_loss', 'mse_loss', 'bce_loss', 'attn_loss', 'loss']
        self.graph_plotter = Graph_Plotting()

    def run(self):
        dir_1 = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_5_seconds'
        file_list_1 = ['train.3.log', 'train.2.log', 'train.log']
        output_dict_1 = self.get_get_loss_from_dir(dir_1, file_list_1)

        dir_2 = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_10_seconds'
        file_list_2 = ['train.5.log', 'train.1.log']
        output_dict_2 = self.get_get_loss_from_dir(dir_2, file_list_2)

        dir_3 = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_15_seconds'
        file_list_3 = ['train.5.log', 'train.4.log', 'train.log']
        output_dict_3 = self.get_get_loss_from_dir(dir_3, file_list_3)

        dir_4 = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_dynamic_20_seconds'
        file_list_4 = ['train.6.log', 'train.5.log', 'train.3.log', 'train.2.log', 'train.1.log', 'train.log']
        output_dict_4 = self.get_get_loss_from_dir(dir_3, file_list_3)



        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/loss_vs_epoch.png'
        x_list = None
        y_list = [output_dict_1['train']['loss'], output_dict_1['valid']['loss'], output_dict_2['train']['loss'], output_dict_2['valid']['loss'], output_dict_3['train']['loss'], output_dict_3['valid']['loss'], output_dict_4['train']['loss'], output_dict_4['valid']['loss']]
        legend_list = ['train_5 train', 'train_5 valid', 'train_10 train', 'train_10 valid', 'train_15 train', 'train_15 valid', 'train_20 train', 'train_20 valid']
        self.graph_plotter.single_plot(fig_file_name, x_list, y_list, legend_list, title=None, x_label='epoch number', y_label='loss')





    def get_get_loss_from_dir(self, dir_name, file_list):
        for i,file_name in enumerate(file_list):
            full_file_name = os.path.join(dir_name, file_name)
            print('reading %s' % full_file_name)
            output_dict = self.get_loss_from_file(full_file_name)
            print('epoch numbers are %i %i' % (output_dict['epoch_numbers'][0], output_dict['epoch_numbers'][1]))
            if i == 0:
                dir_output_dict = copy.deepcopy(output_dict)
            else:
                for k in output_dict['train']:
                    dir_output_dict['train'][k].extend(output_dict['train'][k])
                for k in output_dict['valid']:
                    dir_output_dict['valid'][k].extend(output_dict['valid'][k])
        return dir_output_dict


    def get_loss_from_file(self, file_name):
        '''
        Get a dict of keys and values
        output_dict['train'] = {'iter_time': [7.550,..], 'forward_time':[0.952,..], 'l1_loss':[0.989,..], ...}
        output_dict['valid'] = {...}
        output_dict['epoch_numbers'] = [1, 62]    # First and last epoch number
        '''
        with open(file_name, 'r') as f:
            file_lines = f.readlines()

        useful_lines = []
        for l in file_lines:
            if 'results' in l:
                useful_lines.append(l)
        assert len(useful_lines) > 0, 'List is empty, file name is %s' % file_name

        output_dict = self.init_output_dict(useful_lines)

        for l in useful_lines:
            train_segment_dict, valid_segment_dict = self.get_train_valid_dict_from_line(l)
            for k in train_segment_dict:
                output_dict['train'][k].append(train_segment_dict[k])
            for k in valid_segment_dict:
                output_dict['valid'][k].append(valid_segment_dict[k])

        return output_dict


    def init_output_dict(self, useful_lines):
        output_dict = {}

        e_1 = self.get_epoch_number(useful_lines[0])
        e_2 = self.get_epoch_number(useful_lines[-1])
        output_dict['epoch_numbers'] = [e_1, e_2]

        train_segment_dict, valid_segment_dict = self.get_train_valid_dict_from_line(useful_lines[0])

        output_dict['train'] = {k:[] for k in train_segment_dict}
        output_dict['valid'] = {k:[] for k in valid_segment_dict}

        return output_dict

    def get_train_valid_dict_from_line(self, line):

        header_str, r = line.split('[train]')
        train_str,  r = r.split('[valid]')
        valid_str,  r = r.split('[att_plot]')

        train_segment_dict = self.get_key_value_dict_from_segment(train_str)
        valid_segment_dict = self.get_key_value_dict_from_segment(valid_str)

        return train_segment_dict, valid_segment_dict


    def get_key_value_dict_from_segment(self, line_segment):
        '''
        e.g. line_segment is between [train] and [valid], or [valid] and [att_plot]:
        ... [train] iter_time=7.550, forward_time=0.952, l1_loss=0.989, mse_loss=0.840, bce_loss=0.130, attn_loss=0.004, loss=1.962, backward_time=0.883, optim_step_time=0.011, lr_0=0.001, train_time=9.588, time=1 hour, 19 minutes and 54.41 seconds, total_count=500, [valid] ...
        '''
        segment_dict = {}
        segment_list = line_segment.split(' ')
        for x in segment_list:
            if '=' in x:
                k = x.split('=')[0]
                v = x.split('=')[1].replace(',', '')
                if k != 'time':
                    segment_dict[k] = float(v)

        return segment_dict

    def get_epoch_number(self, l):
        '''
        e.g. line:
        [air209] 2021-10-18 23:44:50,895 (trainer:250) INFO: 1epoch results: [train] iter_time=7.550, forward_time=0.952, l1_loss=0.989, mse_loss=0.840, bce_loss=0.130, attn_loss=0.004, loss=1.962, backward_time=0.883, optim_step_time=0.011, lr_0=0.001, train_time=9.588, time=1 hour, 19 minutes and 54.41 seconds, total_count=500, [valid] l1_loss=0.788, mse_loss=0.512, bce_loss=0.073, attn_loss=0.002, loss=1.375, time=1 minute and 56.46 seconds, total_count=13, [att_plot] time=4.54 seconds, total_count=0
        '''
        line_list = l.split(' ')
        for x in line_list:
            if 'epoch' in x:
                return int(x.replace('epoch',''))

        

class Compute_Feat_Stats(object):
    """
        compute mean variance of acoustic features
        For each file:
            1. compute the "mean vector sequence"
            2. compute the variance of this sequence
            3. record file length
        Then, use file lengths for weighted average
    """
    def __init__(self):
        super(Compute_Feat_Stats, self).__init__()
        self.work_dir = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated'
        self.feat_scp = 'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_5/decode_valid.loss.best/same_[n1]_file_per_speaker_draw_[n2]/eval1/norm/feats.scp' 
        # e.g. exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_5/decode_valid.loss.best/same_11_file_per_speaker_draw_3/eval1/denorm/feats.scp
        # One line of this file:
        #   p024_001 exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_5/decode_valid.loss.best/same_11_file_per_speaker_draw_3/eval1/log/output.1/denorm/p024_001.npy

    def run(self):

        # self.run_single_number(1, 30)

        num_range = [1,31] # 1,31 for 1-30
        num_draw = 30
        var_list = []
        for i in range(num_range[0],num_range[1]):
            mean_var = self.run_single_number(i, num_draw)
            var_list.append(mean_var)

        print(var_list)

    def run_single_number(self, i, num_draw):
        feat_scp_i = self.feat_scp.replace('[n1]',str(i))
        full_feat_scp_i = os.path.join(self.work_dir, feat_scp_i)
        list_file_list = []

        for j in range(num_draw):
            feat_scp_i_j = full_feat_scp_i.replace('[n2]',str(j+1))
            with open(feat_scp_i_j, 'r') as f:
                f_lines = f.readlines()
                file_list = [l.strip().split(' ')[1] for l in f_lines]
            list_file_list.append(file_list)

        num_files = len(list_file_list[0])

        sum_var = 0.
        sum_file_length = 0.

        for k in range(num_files):
            mean_seq, file_length, data_dim = self.compute_mean_seq(list_file_list, k)
            file_sum_var = self.compute_sum_var(list_file_list, k, mean_seq)
            sum_var += file_sum_var
            sum_file_length += file_length

        mean_var = sum_var / (float(num_draw-1) * data_dim * sum_file_length)
        print('var of %i files is %f' % (i, mean_var), flush=True)
        return mean_var

    def compute_mean_seq(self, list_file_list, k):
        data_list = []
        prev_file_length = None
        for file_list in list_file_list:
            file_name = file_list[k]
            full_file_name = os.path.join(self.work_dir, file_name)
            data_file = numpy.load(full_file_name)
            file_length = data_file.shape[0]
            data_dim    = data_file.shape[1]
        
            if prev_file_length is not None:
                assert file_length == prev_file_length
            prev_file_length = file_length

            data_list.append(data_file)

        data_list = numpy.array(data_list)
        data_mean = numpy.mean(data_list, axis=0)

        return data_mean, file_length, data_dim

    def compute_sum_var(self, list_file_list, k, mean_seq):
        sum_var = 0.
        for file_list in list_file_list:
            file_name = file_list[k]
            full_file_name = os.path.join(self.work_dir, file_name)
            data_file = numpy.load(full_file_name)
            diff_data = data_file - mean_seq
            sum_var += numpy.sum(diff_data **2)
        return sum_var

class Compute_Speaker_Embed_Stats(object):
    """
        compute covariance and cosine distance of speaker embeddings
        For each speaker:
            1. compute the "mean vector"
            2. compute the covariance of this speaker
    """
    def __init__(self):
        super(Compute_Speaker_Embed_Stats, self).__init__()
        self.work_dir = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated'
        self.feat_scp = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated/exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_1/decode_valid.loss.best/speaker_draw_1_30_file_per_speaker_draw_30/spk_embed_eval1/spk_embed/feats.scp'
        self.file_name_dict = self.make_file_name_dict()
        self.speaker_id_list = ['p293', 'p210', 'p026', 'p024', 'p313', 'p223', 'p141', 'p386', 'p178', 'p290']
        # One line of this file:
        #   p024_10_13 exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_5/decode_valid.loss.best/speaker_draw_1_30_file_per_speaker_draw_30/spk_embed_eval1/log/output.2/spk_embed/p024_10_13.npy

    def make_file_name_dict(self):
        with open(self.feat_scp, 'r') as f:
            f_lines = f.readlines()
        file_name_dict = {l.strip().split(' ')[0]: l.strip().split(' ')[1] for l in f_lines}
        return file_name_dict

    def run(self):

        num_range = [1,31] # 1,31 for 1-30
        num_draw = 30

        if False:
            var_list = []
            abs_off_diag_list = []
            for i in range(num_range[0],num_range[1]):
                cov_i = self.run_cov_single_number(i, 30)
                var_i = numpy.trace(cov_i)
                var_list.append(var_i)
                abs_off_diag = numpy.sum(numpy.absolute(cov_i)) - numpy.sum(numpy.diag(cov_i))
                abs_off_diag_list.append(abs_off_diag)

            print(var_list)
            print(abs_off_diag_list)

        if True:
            cos_mean_list = []
            cos_std_list  = []
            for i in range(num_range[0],num_range[1]):
                cos_i_list = self.run_cosine_single_number(i, 30)
                cos_mean_list.append(numpy.mean(cos_i_list))
                cos_std_list.append(numpy.std(cos_i_list))

            print(cos_mean_list)
            print(cos_std_list)

    def run_cosine_single_number(self, i, num_draw):
        cosine_distance_list = []
        for speaker_id in self.speaker_id_list:
            file_name = self.file_name_dict['%s_all' % (speaker_id)]
            full_file_name = os.path.join(self.work_dir, file_name)
            all_spk_embed = numpy.load(full_file_name)
            for j in range(1,num_draw+1):
                file_name = self.file_name_dict['%s_%i_%i' % (speaker_id, i, j)]
                full_file_name = os.path.join(self.work_dir, file_name)
                data_file = numpy.load(full_file_name)
                cosine_distance = scipy.spatial.distance.cosine(data_file, all_spk_embed)
                cosine_distance_list.append(cosine_distance)

        cosine_distance_list = numpy.array(cosine_distance_list)
        return cosine_distance_list

    def run_cov_single_number(self, i, num_draw):
        diff_spk_embed_list = []
        for speaker_id in self.speaker_id_list:
            mean_spk_embed = self.compute_mean_spk_embed(speaker_id, i, num_draw)
            for j in range(1,num_draw+1):
                file_name = self.file_name_dict['%s_%i_%i' % (speaker_id, i, j)]
                full_file_name = os.path.join(self.work_dir, file_name)
                data_file = numpy.load(full_file_name)
                diff_spk_embed = data_file - mean_spk_embed
                diff_spk_embed_list.append(diff_spk_embed)

        diff_spk_embed_list = numpy.array(diff_spk_embed_list)
        return numpy.cov(diff_spk_embed_list.T)


            

    def compute_mean_spk_embed(self, speaker_id, i, num_draw):
        spk_embed_list = []
        for j in range(1,num_draw+1):
            file_name = self.file_name_dict['%s_%i_%i' % (speaker_id, i, j)]
            full_file_name = os.path.join(self.work_dir, file_name)
            data_file = numpy.load(full_file_name)
            spk_embed_list.append(data_file)

        spk_embed_list = numpy.array(spk_embed_list)
        mean_spk_embed = numpy.mean(spk_embed_list, axis=0)

        return mean_spk_embed




        # feat_scp_i = self.feat_scp.replace('[n1]',str(i))
        # full_feat_scp_i = os.path.join(self.work_dir, feat_scp_i)
        # list_file_list = []

        # for j in range(num_draw):
        #     feat_scp_i_j = full_feat_scp_i.replace('[n2]',str(j+1))
        #     with open(feat_scp_i_j, 'r') as f:
        #         f_lines = f.readlines()
        #         file_list = [l.strip().split(' ')[1] for l in f_lines]
        #     list_file_list.append(file_list)

        # num_files = len(list_file_list[0])

        # sum_var = 0.
        # sum_file_length = 0.

        # for k in range(num_files):
        #     mean_seq, file_length, data_dim = self.compute_mean_seq(list_file_list, k)
        #     file_sum_var = self.compute_sum_var(list_file_list, k, mean_seq)
        #     sum_var += file_sum_var
        #     sum_file_length += file_length

        # mean_var = sum_var / (float(num_draw-1) * data_dim * sum_file_length)
        # print('var of %i files is %f' % (i, mean_var), flush=True)
        # return mean_var





        
def quick_plot_2():
    import matplotlib.pyplot as plt
    # plt = plt
    fig, ax = plt.subplots()

    x = range(1,31)
    x = [5,10,15,20,30,40,50]
    x = numpy.array(x)

    y_1 = [0.04616709131023122, 0.04601511684944852, 0.045994969194806695, 0.045964920889689696, 0.04594817666802555, 0.045935238308480224, 0.04593388638459146]
    s_1 = [9.550748494316412e-05, 7.70497387087505e-05, 6.091346995302609e-05, 5.035002781802701e-05, 3.321269732694764e-05, 2.2930087178195888e-05, 2.0559180427954127e-05]

    y_2 = [0.047199311055398235, 0.047042848862345436, 0.04702106192138874, 0.046988657890405086, 0.046959622367285195, 0.04695949859477374, 0.04696452143162282]
    s_2 = [0.00012948084368494744, 9.775863224131283e-05, 6.50768248105385e-05, 5.5998723285499264e-05, 4.023725810806004e-05, 3.0121790714987124e-05, 2.4689356762882563e-05]
    y_3 = [0.04784335082086424, 0.04754543270682916, 0.04754409038435876, 0.0475016423731318, 0.04747806830590384, 0.047471661688242524, 0.04746612660302264]
    s_3 = [0.000128391028044271, 0.00012710650438901863, 7.859104468390857e-05, 8.168481437507653e-05, 6.214163830807759e-05, 3.7546626983043835e-05, 3.454785412743308e-05]

    y_4 = [0.04672423207226934, 0.04650719747050768, 0.0464484447781514, 0.0464410118963052, 0.04639430339267063, 0.0463776359285435, 0.046388357064634976]
    s_4 = [0.00017550971573297253, 0.00011986946417787006, 7.039738426918097e-05, 6.521659192214309e-05, 4.741678632619414e-05, 3.5345797779041424e-05, 3.068553259724052e-05]

    y_1 = numpy.array(y_1)
    s_1 = numpy.array(s_1)
    y_2 = numpy.array(y_2)
    s_2 = numpy.array(s_2)
    y_3 = numpy.array(y_3)
    s_3 = numpy.array(s_3)
    y_4 = numpy.array(y_4)
    s_4 = numpy.array(s_4)

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    ax.plot(x, y_1, 'b-', label='train 5 average')
    ax.fill(numpy.concatenate([x, x[::-1]]),
             numpy.concatenate([y_1 - 1.9600 * s_1,
                            (y_1 + 1.9600 * s_1)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence')
    ax.plot(x, y_2, 'r-', label='train_10 average')
    ax.fill(numpy.concatenate([x, x[::-1]]),
             numpy.concatenate([y_2 - 1.9600 * s_2,
                            (y_2 + 1.9600 * s_2)[::-1]]),
             alpha=.5, fc='r', ec='None', label='95% confidence')
    ax.plot(x, y_3, 'g-', label='train 15 average')
    ax.fill(numpy.concatenate([x, x[::-1]]),
             numpy.concatenate([y_3 - 1.9600 * s_3,
                            (y_3 + 1.9600 * s_3)[::-1]]),
             alpha=.5, fc='g', ec='None', label='95% confidence')
    ax.plot(x, y_4, 'y-', label='train_20 average')
    ax.fill(numpy.concatenate([x, x[::-1]]),
             numpy.concatenate([y_4 - 1.9600 * s_4,
                            (y_4 + 1.9600 * s_4)[::-1]]),
             alpha=.5, fc='y', ec='None', label='95% confidence')
    ax.set_xlabel('num_seconds_adaptation_inference')
    ax.set_ylabel('MSE')

    ax.legend()

    save_dir = '/home/dawna/tts/mw545/Export_Temp'
    fig_file_name = os.path.join(save_dir, 'mse_with_var.png')

    print('Saving to %s' % fig_file_name)
    fig.savefig(fig_file_name, format="png")
    plt.close(fig)

def quick_plot_3():
    import matplotlib.pyplot as plt
    # plt = plt
    fig, ax = plt.subplots()

    # x = range(1,31)
    x = [5,10,15,20,30,40,50]
    x = numpy.array(x)
    y_1 = [0.04719931119561402, 0.04704284900825263, 0.047021061941826106, 0.046988657775024575, 0.046959622296401195, 0.04695949862504171, 0.04696452144611006]
    s_1 = [0.0001294809231001935, 9.775913347728959e-05, 6.50769505760014e-05, 5.599873396815888e-05, 4.0237345492345746e-05, 3.012177825656241e-05, 2.4689396531922874e-05]
    y_2 = [0.04789288308859493, 0.04758192871309197, 0.047544088465575556, 0.047501642346744304, 0.04747806809273445, 0.047471661552424646, 0.04746612643021055]
    s_2 = [0.00017174586693779654, 0.00012058878028336817, 7.858847156629472e-05, 8.168473836435139e-05, 6.214159883551747e-05, 3.7546575861205234e-05, 3.4547758034071135e-05]

    
    y_1 = numpy.array(y_1)
    s_1 = numpy.array(s_1)
    y_2 = numpy.array(y_2)
    s_2 = numpy.array(s_2)

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    ax.plot(x, y_1, 'b-', label='train_10 average')
    # ax.fill(numpy.concatenate([x, x[::-1]]),
    #          # numpy.concatenate([y_1 - 1.9600 * s_1,
    #                         (y_1 + 1.9600 * s_1)[::-1]]),
    #          alpha=.5, fc='b', ec='None', label='95% confidence')
    ax.plot(x, y_2, 'r-', label='train_15 average')
    # ax.fill(numpy.concatenate([x, x[::-1]]),
    #          numpy.concatenate([y_2 - 1.9600 * s_2,
    #                         (y_2 + 1.9600 * s_2)[::-1]]),
    #          alpha=.5, fc='r', ec='None', label='95% confidence')
    ax.set_xlabel('num_seconds_adaptation_inference')
    ax.set_ylabel('Cosine Distance')

    ax.legend()

    save_dir = '/home/dawna/tts/mw545/Export_Temp'
    fig_file_name = os.path.join(save_dir, 'cosine_no_var.png')
    fig.savefig(fig_file_name, format="png")
    plt.close(fig)


if __name__ == '__main__':

    dv_y_cfg = dv_y_configuration()
    # process_runner = Make_Corpus(dv_y_cfg)
    # process_runner = Make_Data(dv_y_cfg)
    # process_runner = Make_X_Vector(dv_y_cfg)
    process_runner = Make_Spk_Embed_Model_Data(dv_y_cfg)

    # process_runner = Make_dir_listening_test()
    # process_runner = Make_loss_plot()
    # process_runner = ESPNet_log_reader_plotter()
    # process_runner = Compute_Feat_Stats()
    # process_runner = Compute_Speaker_Embed_Stats()

    process_runner.run()

    # write_train_valid_test_file_id_list(dv_y_cfg)
    # setup_integrated_exp_directory(dv_y_cfg)
    # temp_make_cmp_x_vector_5()
    # quick_plot()

    # remove_norm_denorm_from_exp_directory()
    
    pass






#############################################

def collect_tokens():
    '''
    This is a check function before making mono labels
    Collect all tokens in the current file
    See if we can do a mapping to the target tokens
    '''
    dv_y_cfg = dv_y_configuration()

    lab_state_align_dir = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/lab/labels'
    speaker_id_list = dv_y_cfg.speaker_id_list_dict['all']
    
    token_list = []
    for speaker_id in speaker_id_list:
        for file_id in dv_y_cfg.spk_file_list[speaker_id]:
            lab_file_name = os.path.join(lab_state_align_dir, file_id+'.lab')

            with open(lab_file_name,'r') as f:
                f_lines = f.readlines()

            for l in f_lines:
                t = l.split('-')[1].split('+')[0]
                if t not in token_list:
                    token_list.append(t)
        token_list.sort()

    print(token_list)
    print(len(token_list))

def make_mono_labs():
    '''
    Example mono label file:
      /home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/vctk/tts_xvector/downloads/VCTK-Corpus/lab/mono/p225/p225_001.lab

    Current state-aligned label file:
      /data/vectra2/tts/mw545/Data/Data_Voicebank_48kHz/label/label_state_align/p225_001.lab

    Paused and skipped:
      mono-lab files seem to be used to determine silence only, to write segments files
    '''
    dv_y_cfg = dv_y_configuration()

    lab_state_align_dir = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/lab/labels'
    lab_mono_dir = '/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/lab/mono'

    speaker_id_list = dv_y_cfg.speaker_id_list_dict['all']
    for speaker_id in speaker_id_list:
        # Create speaker directory in 
        speaker_dir = os.path.join(lab_mono_dir, speaker_id)
        prepare_script_file_path(speaker_dir)
        pass
