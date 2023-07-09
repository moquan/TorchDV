######################################################
# make_data.py
# Data Prep for ESPNet 2 Tacotron 2 + CUED Voicebank #
# Note:                                              #
#   Some functions here write bash scripts to run    #
#   Run or submit after writing                      #
######################################################

import os, pickle, numpy
import subprocess
from frontend_mw545.modules import prepare_script_file_path, read_file_list, File_List_Selecter, List_Random_Loader
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
        self.make_wav24()

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
        for dir_name in ['tr_no_dev', 'dev', 'eval1']:
            full_dir_name = os.path.join(self.data_dir, dir_name)
            prepare_script_file_path(full_dir_name)

            if dir_name == 'tr_no_dev':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'tr_no_dev.scp')
            if dir_name == 'dev':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'valid.scp')
            if dir_name == 'eval1':
                file_id_list_file = os.path.join(dv_y_cfg.file_id_list_dir,'test.scp')

            # self.make_segments(full_dir_name, file_id_list_file)
            # self.make_spk2utt(full_dir_name, file_id_list_file)
            # self.make_text(full_dir_name, file_id_list_file)
            # self.make_utt2spk(full_dir_name, file_id_list_file)
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

    def run(self):
        # spk_embed_file = '/home/dawna/tts/mw545/TorchDV/dv_cmp_baseline/dvy_cmp_lr1E-04_fpu40_LRe512L_LRe512L_Lin512L_DV512S1B161T40D3440/dv_spk_dict.dat'
        # spk_embed_name = 'cmp'
        spk_embed_file = '/home/dawna/tts/mw545/TorchDV/dv_wav_sincnet/dvy_wav_lr1E-04_fpu4_Sin80_LRe512L_LRe512L_Lin512L_DV512S1B161M77T160/dv_spk_dict.dat'
        spk_embed_name = 'sincnet'

        spk_embed_values = pickle.load(open(spk_embed_file, 'rb'))
        exp_dir_name = os.path.join(self.x_vector_dir, spk_embed_name)

        self.save_speaker_embed_files(spk_embed_values, exp_dir_name)

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
        spk_model_data_name = 'cmp'

        exp_dir_name = os.path.join(self.spk_model_data_dir, spk_model_data_name)

        if spk_model_data_name == 'cmp':
            self.cmp_data_dir = os.path.join(exp_dir_name, 'cmp_data_dir')
            # prepare_script_file_path(self.cmp_data_dir)
            # self.prepare_cmp_dir(exp_dir_name)

        for dir_name in ['tr_no_dev', 'dev', 'eval1']:
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
            self.write_random_n_file_cmp_scp(full_dir_name, file_id_list_file, file_id_list_file_SR, 5)

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
                file_id_list  = numpy.random.choice(file_id_list_SR_dict[speaker_id], n, replace=False)
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



def temp_change():
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

# temp_change()




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
        self.file_id_list = ['p024_001','p026_001','p141_001','p178_001','p290_001']
        self.sample_tool_dir = '/home/dawna/tts/mw545/tools/weblisteningtest'
        self.sample_root_dir = os.path.join(self.sample_tool_dir, 'samples-mw545')

    def run(self):
        self.file_dir_type = {}
        self.file_dir_type['orig'] = ['/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/data_speaker_206/downloads/VCTK-Corpus/wav24','speaker_id/file_id.wav']

        self.setting_1()

        self.make_dirs()
        self.write_command()

    def make_file_dir_type_2_stage(self):
        file_dir_type = {}
        work_dir = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage'

        # Griffin-Lim valid_best exps; poor quality, almost useless
        file_dir_type['vocoder_gl'] = [os.path.join(work_dir, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp/decode_valid.loss.best/eval1/wav'), 'file_id.wav']
        file_dir_type['sincnet_gl'] = [os.path.join(work_dir, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_sincnet/decode_valid.loss.best/eval1/wav'), 'file_id.wav']
        file_dir_type['xvector_gl'] = [os.path.join(work_dir, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_xvector/decode_valid.loss.best/eval1/wav'), 'file_id.wav']

        file_dir_type['vocoder_train'] = [os.path.join(work_dir, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp/decode_train.loss.best/eval1/wav_pwg'), 'file_id_gen.wav']
        file_dir_type['sincnet_train'] = [os.path.join(work_dir, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_sincnet/decode_train.loss.best/eval1/wav_pwg'), 'file_id_gen.wav']
        file_dir_type['xvector_train'] = [os.path.join(work_dir, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_xvector/decode_train.loss.best/eval1/wav_pwg'), 'file_id_gen.wav']

        file_dir_type['vocoder_valid'] = [os.path.join(work_dir, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp/decode_valid.loss.best/eval1/wav_pwg'), 'file_id_gen.wav']
        file_dir_type['sincnet_valid'] = [os.path.join(work_dir, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_sincnet/decode_valid.loss.best/eval1/wav_pwg'), 'file_id_gen.wav']
        file_dir_type['xvector_valid'] = [os.path.join(work_dir, 'exp/tts_train_raw_phn_tacotron_g2p_en_no_space_xvector/decode_valid.loss.best/eval1/wav_pwg'), 'file_id_gen.wav']

        return file_dir_type

    def make_file_dir_type_integrated_cmp(self):
        file_dir_type = {}
        work_dir = '/home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated'

        # "Same" means adaptation and synthesis are same data, almost useless
        file_dir_type['vocoder_same_train'] = [os.path.join(work_dir, 'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_same/decode_train.loss.best/eval1/wav_pwg'), 'file_id_gen.wav']
        file_dir_type['vocoder_same_valid'] = [os.path.join(work_dir, 'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_same/decode_valid.loss.best/eval1/wav_pwg'), 'file_id_gen.wav']

        file_dir_type['vocoder_random_1_train'] = [os.path.join(work_dir, 'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_1/decode_train.loss.best/eval1/wav_pwg'), 'file_id_gen.wav']
        file_dir_type['vocoder_random_1_valid'] = [os.path.join(work_dir, 'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_1/decode_valid.loss.best/eval1/wav_pwg'), 'file_id_gen.wav']

        file_dir_type['vocoder_random_5_train'] = [os.path.join(work_dir, 'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_5/decode_train.loss.best/eval1/wav_pwg'), 'file_id_gen.wav']
        file_dir_type['vocoder_random_5_valid'] = [os.path.join(work_dir, 'exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_5/decode_valid.loss.best/eval1/wav_pwg'), 'file_id_gen.wav']


        return file_dir_type

    def setting_2(self):
        self.table_name = 'Voicebank_ESPNet2_Tacotron2_vocoder_2_stage_vs_integrated'
        self.title_list = ['orig','vocoder_2_stage', 'vocoder_integrated_random_5']

        file_dir_type_2_stage = self.make_file_dir_type_2_stage()
        file_dir_type_integrated = self.make_file_dir_type_integrated_cmp()

        self.file_dir_type['vocoder_2_stage'] = file_dir_type_2_stage['vocoder_valid']
        self.file_dir_type['vocoder_integrated_random_5'] = file_dir_type_integrated['vocoder_random_5_valid']

    def setting_1(self):
        self.table_name = 'Voicebank_ESPNet2_Tacotron2_vocoder_integrated_1_5'
        self.title_list = ['orig','vocoder_integrated_random_1', 'vocoder_integrated_random_5']

        file_dir_type_2_stage = self.make_file_dir_type_2_stage()
        file_dir_type_integrated = self.make_file_dir_type_integrated_cmp()

        self.file_dir_type['vocoder_integrated_random_1'] = file_dir_type_integrated['vocoder_random_1_valid']
        self.file_dir_type['vocoder_integrated_random_5'] = file_dir_type_integrated['vocoder_random_5_valid']


    def make_date_str(self):
        '''
        yyyy_mm_dd format
        '''
        from datetime import datetime
        return datetime.today().strftime('%Y_%m_%d')

    def make_dirs(self):
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

    def make_file_name(self, file_id, file_dir_type):
        dir_name  = file_dir_type[0]
        file_type = file_dir_type[1]
        new_file_name = file_type.replace('file_id',file_id)
        if 'speaker_id' in file_type:
            s = file_id.split('_')[0]
            new_file_name = new_file_name.replace('speaker_id',s)

        new_file_name_full = os.path.join(dir_name, new_file_name)
        return new_file_name_full

    def write_command(self):
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

            f.write('python demotable_html_wav.py --idexp "^(p[0-9]+_[0-9]+)"  --tablename %s \\ \n' % self.table_name)
            dir_list = [self.sample_dir + '/' + t for t in self.title_list]
            f.write(' '.join(dir_list) + ' \\ \n')
            f.write('--titles %s \n' %(' '.join(self.title_list)))

            date_str = self.make_date_str()
            weblink_dir = os.path.join(self.sample_tool_dir, 'Samples_Webpage_Dir', date_str+ '_' + self.table_name)
            f.write('mkdir %s \n' % weblink_dir)
            f.write('mv %s %s.html %s/ \n' % (self.table_name, self.table_name, weblink_dir))
            f.write('\n'*2)
            f.write('Add this line to %s/Samples_Webpage_Dir/2021_07_17_Voicebank_ESPNet2_Tacotron2.html: \n' % self.sample_tool_dir)
            # f.write('<li><a href="samples-mw545/%s/%s.html">%s</a> </li>\n' % (date_str+ '_' + self.table_name, self.table_name, self.table_name))
            f.write('<li><a href="%s/%s.html">%s</a> </li>\n' % (date_str+ '_' + self.table_name, self.table_name, self.table_name))

            f.write('\nGo back to %s ? \n\n' % os.path.dirname(os.path.realpath(__file__)))

        print('cat %s \n' % run_script)



if __name__ == '__main__':

    dv_y_cfg = dv_y_configuration()
    # process_runner = Make_Corpus(dv_y_cfg)
    # process_runner = Make_Data(dv_y_cfg)
    # process_runner = Make_X_Vector(dv_y_cfg)
    process_runner = Make_dir_listening_test()
    # process_runner = Make_Spk_Embed_Model_Data(dv_y_cfg)
    process_runner.run()

    # write_train_valid_test_file_id_list(dv_y_cfg)
    # setup_integrated_exp_directory(dv_y_cfg)
    
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
