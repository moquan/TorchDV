# frontend_tests.py

import os, sys, pickle, time, datetime, shutil, logging, copy 
import math, numpy, scipy
numpy.random.seed(545)

from frontend_mw545.modules import make_logger, read_file_list, prepare_script_file_path
from frontend_mw545.modules import Data_Replicate_Test, Graph_Plotting, Build_Log_File_Reader
from frontend_mw545.data_io import Data_File_IO


class Data_List_File_IO_Test(object):
    """docstring for Data_List_File_IO_Test"""
    def __init__(self, cfg=None):
        super(Data_List_File_IO_Test, self).__init__()
        self.cfg = cfg
        self.logger = make_logger("Data_List_File_IO_Test")

        self.cfg = cfg

        from frontend_mw545.data_io import Data_List_File_IO
        # print(file_id_list)
        self.DLFIO = Data_List_File_IO(cfg)

    def test(self):
        self.test_dv_used()

    def run(self):
        self.DLFIO.write_file_list_compute_norm_info()

    def test_cfg_used(self):
        '''
        Make new files and compare with old files
        Result: Good
        '''
        cfg = self.cfg
        in_file_name = cfg.file_id_list_file['all']
        out_file_name = cfg.file_id_list_file['used'] + '.test'
        not_used_file_name = cfg.file_id_list_file['excluded'] + '.test'
        self.DLFIO.write_file_list_cfg_used(in_file_name, out_file_name, not_used_file_name)

        if self.check_file_list_files(cfg.file_id_list_file['used'], out_file_name):
            self.logger.info('Files Matched!')
        if self.check_file_list_files(cfg.file_id_list_file['excluded'], not_used_file_name):
            self.logger.info('Files Matched!')

    def test_dv_used(self):
        '''
        Make new files and compare with old files
        Result: Good; exactly same files
        '''
        cfg = self.cfg

        in_file_name = cfg.file_id_list_file['all']
        out_file_name = cfg.file_id_list_file['dv_test'] + '.test'

        self.DLFIO.write_file_list_dv_test(in_file_name, out_file_name)

        if self.check_file_list_files(cfg.file_id_list_file['dv_test'], out_file_name):
            self.logger.info('Files Matched!')

    def check_file_list_files(self, file_1, file_2):
        '''
        1. check length
        2. check content and order
        3. check content in wrong order
            3.1 sort by speaker first, to reduce amount of searching
            3.2 sort for each speaker, then compare
        '''

        file_list_1 = self.DLFIO.read_file_list(file_1)
        file_list_2 = self.DLFIO.read_file_list(file_2)

        l_1 = len(file_list_1)
        l_2 = len(file_list_2)

        if l_1 != l_2:
            self.logger.info('Difference lengths %i %i '%(l_1, l_2))
            self.logger.info('%s %s'% (file_1, file_2))
            return False
        else:
            l = l_1

        all_match = True
        for i in range(l):
            if file_list_1[i] != file_list_2[i]:
                self.logger.info('Difference files at line %i, %s %s'%(i, file_list_1[i], file_list_2[i]))
                all_match = False
                break
        if all_match:
            # 2 are exactly the same
            return True

        from frontend_mw545.modules import File_List_Selecter
        FL_Selecter = File_List_Selecter()
        speaker_id_list = self.cfg.speaker_id_list_dict['all']
        file_dict_1 = FL_Selecter.sort_by_speaker_list(file_list_1, speaker_id_list)
        file_dict_2 = FL_Selecter.sort_by_speaker_list(file_list_2, speaker_id_list)

        for speaker_id in speaker_id_list:
            l_1 = len(file_dict_1[speaker_id])
            l_2 = len(file_dict_2[speaker_id])
            if l_1 != l_2:
                self.logger.info('Difference lengths of speaker %s, %i %i '%(speaker_id, l_1, l_2))
                self.logger.info('%s %s'% (file_1, file_2))
                return False

            file_dict_1[speaker_id].sort()
            file_dict_2[speaker_id].sort()

            if file_dict_1[speaker_id] != file_dict_2[speaker_id]:
                self.logger.info('Difference files of speaker %s, %i %i '%(speaker_id, l_1, l_2))
                self.logger.info('%s %s'% (file_dict_1[speaker_id], file_dict_2[speaker_id]))
                return False

        # All speakers are matched, but orders may be different
        self.logger.info('Same files, different orders')
        return True




class Data_File_Converter_Test(object):
    """docstring for Data_File_Converter_Test"""
    def __init__(self, cfg=None):
        super(Data_File_Converter_Test, self).__init__()
        self.logger = make_logger("DFC_Test")

        self.cfg = cfg
        self.DF_IO = Data_File_IO(cfg)
        self.graph_plotter = Graph_Plotting()

        from frontend_mw545.data_converter import Data_File_Converter, Data_File_List_Converter
        self.DFC = Data_File_Converter(cfg)

    def wav_cmp_test(self):
        '''
        Check if the current wav_cmp is consistent with a new generated file
        '''
        cfg = self.cfg
        file_id_list = ['p290_028']

        input_dir  = cfg.wav_dir
        output_dir = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/temp_test_dir'
        wav_cmp_dir = self.cfg.nn_feat_dirs['wav']

        for file_id in file_id_list:
            x = os.path.join(cfg.wav_dir, file_id + '.wav')
            y = os.path.join(output_dir, file_id + '.wav')
            r = os.path.join(wav_cmp_dir, file_id + '.wav')

            self.DFC.wav_2_wav_cmp(x, y)
            y_data, y_num_frame = self.DF_IO.load_data_file_frame(y, 1)
            r_data, r_num_frame = self.DF_IO.load_data_file_frame(r, 1)

            if y_num_frame == r_num_frame:
                print((y_data==r_data).all())
        
    def pitch_16kHz_test(self):
        '''
        Plot pitch scatter and pitch_16kHz on the same graph
        Expect to see "step-up and slope-down" in pitch distance
        '''
        file_id_list = ['p290_028']

        plot_dir = '/home/dawna/tts/mw545/Export_Temp/PNG_out/pitch_16kHz'

        for file_id in file_id_list:
            pitch_reaper_file = os.path.join(self.cfg.pitch_dir, file_id + '.pitch')
            wav_cmp_file      = os.path.join(self.cfg.nn_feat_dirs['wav'], file_id + '.wav')
            pitch_16kHz_file  = os.path.join(self.cfg.nn_feat_dirs['pitch'], file_id + '.pitch')

            self.DFC.pitch_text_2_pitch_1D(pitch_reaper_file, wav_cmp_file, pitch_16kHz_file)

            pitch_reaper_data = self.DF_IO.read_pitch_reaper(pitch_reaper_file)
            pitch_16kHz_data, l_16k = self.DF_IO.load_data_file_frame(pitch_16kHz_file, 1)
            wav_16kHz_data, l_16k = self.DF_IO.load_data_file_frame(wav_cmp_file, 1)

            fig_file_name = os.path.join(plot_dir, file_id+'_pitch_16k.png')
            x_16k = numpy.arange(l_16k) / float(16000) + float(1/32000.) # +1/32000, at the centre of the frames
            pitch_y = numpy.zeros(len(pitch_reaper_data))

            x_list = [x_16k, pitch_reaper_data]
            y_list = [pitch_16kHz_data, pitch_y]
            legend_list = ['pitch_16kHz', 'pitch REAPER']

            self.logger.info('Saving figure to %s' % fig_file_name)
            self.graph_plotter.one_line_one_scatter(fig_file_name, x_list, y_list, legend_list)

            # Make smaller plots of 100 samples each
            num_samples_per_plot = 1000
            num_plots = int(l_16k/(num_samples_per_plot-1))+1

            if False:
                # pitch_16kHz and pitch_reaper
                for i in range(num_plots):
                    start_i = i * num_samples_per_plot
                    end_i   = (i+1) * num_samples_per_plot
                    x_16k_i = x_16k[start_i:end_i]
                    pitch_16kHz_data_i = pitch_16kHz_data[start_i:end_i]

                    start_t = start_i / float(16000)
                    end_t   = end_i / float(16000)
                    pitch_reaper_data_i = []
                    for pitch_t in pitch_reaper_data:
                        if pitch_t >= start_t:
                            if pitch_t <= end_t:
                                pitch_reaper_data_i.append(pitch_t)
                    pitch_y_i = numpy.zeros(len(pitch_reaper_data_i))

                    x_list = [x_16k_i, pitch_reaper_data_i]
                    y_list = [pitch_16kHz_data_i, pitch_y_i]
                    legend_list = ['pitch_16kHz', 'pitch REAPER']

                    fig_file_name = os.path.join(plot_dir, '%s_pitch_16k_%03i.png' % (file_id, i))
                    self.logger.info('Saving figure to %s' % fig_file_name)
                    self.graph_plotter.one_line_one_scatter(fig_file_name, x_list, y_list, legend_list)

            if True:
                # wav_16kHz and pitch_reaper
                for i in range(num_plots):
                    start_i = i * num_samples_per_plot
                    end_i   = (i+1) * num_samples_per_plot
                    x_16k_i = x_16k[start_i:end_i]
                    wav_16kHz_data_i = wav_16kHz_data[start_i:end_i]

                    start_t = start_i / float(16000)
                    end_t   = end_i / float(16000)
                    pitch_reaper_data_i = []
                    for pitch_t in pitch_reaper_data:
                        if pitch_t >= start_t:
                            if pitch_t <= end_t:
                                pitch_reaper_data_i.append(pitch_t)
                    pitch_y_i = numpy.zeros(len(pitch_reaper_data_i))

                    x_list = [x_16k_i, pitch_reaper_data_i]
                    y_list = [wav_16kHz_data_i, pitch_y_i]
                    legend_list = ['wav_16kHz', 'pitch REAPER']

                    fig_file_name = os.path.join(plot_dir, '%s_pitch_wav_16k_%03i.png' % (file_id, i))
                    self.logger.info('Saving figure to %s' % fig_file_name)
                    self.graph_plotter.one_line_one_scatter(fig_file_name, x_list, y_list, legend_list)

    def f0_upsample_test(self):
        '''
        Upsample from 200Hz to 16kHz
        '''
        file_id_list = ['p290_028']

        f0_200Hz_dir  = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/nn_f0200'
        f0_16kHz_dir  = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/nn_f016k'
        plot_dir = '/home/dawna/tts/mw545/Export_Temp/PNG_out/f0_interp_test'

        for file_id in file_id_list:
            f0_200Hz_file  = os.path.join(f0_200Hz_dir, file_id + '.f0')
            f0_16kHz_file  = os.path.join(f0_16kHz_dir, file_id + '.f0')

            self.DFC.f0_200Hz_2_f0_16kHz(f0_200Hz_file, f0_16kHz_file)

            f0_200Hz_data, l_200 = self.DF_IO.load_data_file_frame(f0_200Hz_file, 1)
            f0_16kHz_data, l_16k = self.DF_IO.load_data_file_frame(f0_16kHz_file, 1)

            fig_file_name = os.path.join(plot_dir, file_id+'_f0_upsample.png')

            x_200 = numpy.arange(l_200) / float(200)
            x_16k = numpy.arange(l_16k) / float(16000)

            x_list = [x_200, x_16k]
            y_list = [f0_200Hz_data, f0_16kHz_data+20]
            legend_list = ['200Hz', '16kHz (+20)']

            self.logger.info('Saving figure to %s' % fig_file_name)
            self.graph_plotter.single_plot(fig_file_name, x_list, y_list, legend_list)

    def f0_interpolate_test(self):
        '''
        Plot f0 before/after interpolation
        '''
        file_id_list = ['p290_028']

        f0_reaper_dir = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/reaper_16kHz/f0'
        f0_200Hz_dir  = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/nn_f0200'
        plot_dir = '/home/dawna/tts/mw545/Export_Temp/PNG_out/f0_interp_test'

        for file_id in file_id_list:
            f0_reaper_file = os.path.join(f0_reaper_dir, file_id + '.f0')
            f0_200Hz_file  = os.path.join(f0_200Hz_dir, file_id + '.f0')

            self.DFC.f0_reaper_2_f0_200Hz(f0_reaper_file, f0_200Hz_file)

            f0_reaper_data = self.DF_IO.read_f0_reaper(f0_reaper_file)
            f0_200Hz_data, l = self.DF_IO.load_data_file_frame(f0_200Hz_file, 1)

            fig_file_name = os.path.join(plot_dir, file_id+'_f0_interp.png')
            x = numpy.arange(l) / float(200)

            x_list = [x, x]
            y_list = [f0_reaper_data, f0_200Hz_data+20]
            legend_list = ['reaper_output', 'interpolate (+20)']

            self.logger.info('Saving figure to %s' % fig_file_name)
            self.graph_plotter.single_plot(fig_file_name, x_list, y_list, legend_list)

    def pitch_reaper_test(self):
        '''
        Plot pitch with wav data
        See if pitch locations are sensible
        '''
        file_id_list_file = self.cfg.file_id_list_file['used']
        self.logger.info('Reading file list from %s' % file_id_list_file)
        file_id_list = read_file_list(file_id_list_file)

        # Draw a few files for testing
        list_draw = numpy.random.choice(file_id_list, 100, replace=False)
        file_id_list = list_draw
        # file_id_list = ['p290_028']

        pitch_reaper_dir = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/reaper_16kHz/pitch'
        plot_dir = '/home/dawna/tts/mw545/Export_Temp/PNG_out/pitch_test'

        for file_id in file_id_list:
            pitch_reaper_file = os.path.join(pitch_reaper_dir, file_id + '.pitch')
            pitch_list = self.DF_IO.read_pitch_reaper(pitch_reaper_file)
            wav_data = self.DF_IO.read_wav_2_wav_1D_data(file_id=file_id)

            fig_file_name = os.path.join(plot_dir, file_id+'_pitch_wav.png')
            wav_x = numpy.arange(wav_data.shape[0]) / float(self.cfg.wav_sr)
            pitch_y = numpy.ones(len(pitch_list)) * 8500.

            x_list = [wav_x, pitch_list]
            y_list = [wav_data, pitch_y]
            legend_list = ['wav l='+str(wav_data.shape[0]), 'pitch']

            self.logger.info('Saving figure to %s' % fig_file_name)
            self.graph_plotter.one_line_one_scatter(fig_file_name, x_list, y_list, legend_list)

    def f0_reaper_pml_test(self):
        '''
        Compare f0/lf0 produced by reaper and pml
        '''
        file_id_list_file = self.cfg.file_id_list_file['used']
        self.logger.info('Reading file list from %s' % file_id_list_file)
        file_id_list = read_file_list(file_id_list_file)

        # Draw a few files for testing
        list_draw = numpy.random.choice(file_id_list, 100, replace=False)
        file_id_list = list_draw

        # file_id_list = ['p290_028']

        f0_reaper_dir = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/reaper_16kHz/f0'
        plot_dir = '/home/dawna/tts/mw545/Export_Temp/PNG_out/f0_test'

        for file_id in file_id_list:
            f0_reaper_file = os.path.join(f0_reaper_dir, file_id + '.f0')
            f0_reaper_data = self.DF_IO.read_f0_reaper(f0_reaper_file)
            lf0_pml_data   = self.DF_IO.read_pml_by_name_feat(file_id, 'lf0')
            f0_pml_data    = numpy.exp(lf0_pml_data)

            l_reaper = len(f0_reaper_data)
            l_pml = lf0_pml_data.shape[0]

            l_min = min(l_reaper, l_pml)
            f_diff = numpy.zeros(l_min)

            for i in range(l_min):
                f_r  = f0_reaper_data[i]
                f_p = f0_pml_data[i]

                # check vuv
                if f_r > 0:
                    f_diff[i] = f_r - f_p

            # print(f_diff)
            # print(min(f_diff))
            # print(max(f_diff))
            fig_file_name = os.path.join(plot_dir, file_id+'_f0.png')
            x_r = range(l_reaper)
            x_p = range(l_pml)
            x_list = [x_r, x_p]
            y_list = [f0_reaper_data, f0_pml_data]
            legend_list = ['REAPER l='+str(l_reaper), 'PML l='+str(l_pml)]
            
            self.logger.info('Saving figure to %s' % fig_file_name)
            self.graph_plotter.single_plot(fig_file_name, x_list, y_list, legend_list)

    def vocoder_test_1(self):
        '''
        Need to run in Python 2.7! 
        DFC_test: convert wav file to vocoders then reconstruct wav file
        copy original and save new wav files to temp dir; check by listening
        '''
        
        cfg = self.DFC.cfg
        # test_temp_dir = "/scratch/tmp-mw545/voicebank_208_speakers/test_temp"
        test_temp_dir = "/home/dawna/tts/mw545/Export_Temp/voicebank_208_speakers/vocoder_test"
        if not os.path.exists(test_temp_dir): os.makedirs(test_temp_dir)
        file_id_list = ['p290_028']

        for file_id in file_id_list:
            wav_file = os.path.join(cfg.wav_dir, file_id + '.wav')
            wav_file_1 = os.path.join(test_temp_dir, file_id + '_ori.wav')
            wav_file_2 = os.path.join(test_temp_dir, file_id + '_new.wav')
            vocoder_file_dict = {}
            for feat_name in cfg.acoustic_features:
                vocoder_file_dict[feat_name] = os.path.join(test_temp_dir, file_id + cfg.acoustic_file_ext_dict[feat_name])

            shutil.copyfile(wav_file, wav_file_1)
            self.DFC.wav_2_acoustic(wav_file_1, vocoder_file_dict)
            self.DFC.acoustic_2_wav(vocoder_file_dict, wav_file_2)

    def vocoder_cmp_test(self):
        '''
        Need to run in Python 2.7! 
        DFC_test: convert wav file to vocoders then cmp, then reconstruct wav file
        copy original and save new wav files to temp dir; check by listening
        '''
        cfg = self.DFC.cfg
        # test_temp_dir = "/scratch/tmp-mw545/voicebank_208_speakers/test_temp"
        test_temp_dir = "/home/dawna/tts/mw545/Export_Temp/voicebank_208_speakers/vocoder_cmp_test"
        if not os.path.exists(test_temp_dir): os.makedirs(test_temp_dir)
        file_id_list = ['p290_028']

        for file_id in file_id_list:
            wav_file = os.path.join(cfg.wav_dir, file_id + '.wav')
            wav_file_1 = os.path.join(test_temp_dir, file_id + '_ori.wav')
            wav_file_2 = os.path.join(test_temp_dir, file_id + '_new.wav')
            cmp_file   = os.path.join(test_temp_dir, file_id + '.cmp')
            vocoder_file_dict_1 = {}
            vocoder_file_dict_2 = {}
            for feat_name in cfg.acoustic_features:
                vocoder_file_dict_1[feat_name] = os.path.join(test_temp_dir, file_id + cfg.acoustic_file_ext_dict[feat_name])
                vocoder_file_dict_2[feat_name] = os.path.join(test_temp_dir, file_id + cfg.acoustic_file_ext_dict[feat_name] + '.new')

            shutil.copyfile(wav_file, wav_file_1)
            self.DFC.wav_2_acoustic(wav_file_1, vocoder_file_dict_1)
            self.DFC.pml_2_cmp(vocoder_file_dict_1, cmp_file)
            self.DFC.cmp_2_pml(cmp_file, vocoder_file_dict_2)
            self.DFC.acoustic_2_wav(vocoder_file_dict_2, wav_file_2)

    def wav_1D_data_test(self):
        '''
        Load wav file into numpy array
        Save back to wav file
        check by listening
        '''
        cfg = self.cfg
        # test_temp_dir = "/scratch/tmp-mw545/voicebank_208_speakers/test_temp"
        test_temp_dir = "/home/dawna/tts/mw545/Export_Temp/voicebank_208_speakers/wav_1D_data_test"
        if not os.path.exists(test_temp_dir): os.makedirs(test_temp_dir)
        file_id_list = ['p290_028']
        

        for file_id in file_id_list:
            wav_file = os.path.join(cfg.wav_dir, file_id + '.wav')
            wav_file_1 = os.path.join(test_temp_dir, file_id + '_ori.wav')
            wav_file_2 = os.path.join(test_temp_dir, file_id + '_new.wav')

            shutil.copyfile(wav_file, wav_file_1)
            wav_1D_data = self.DF_IO.read_wav_2_wav_1D_data(wav_file_1)
            self.DF_IO.write_wav_1D_data_2_wav(wav_1D_data, wav_file_2, sample_rate=self.cfg.wav_sr)

    def cmp_data_test_v1(self):
        '''
        Generate cmp file
        Compare with current cmp file; log file name if not equal
        Remove acoustic and cmp files, if equal

        Result:
        Around 10%  files have slightly different cmp files
        But, the PML files can be very very different
        Hypothesis: current PML and cmp files are not compatible
        
        Action:
        New test, cmp_data_test_v2, check if current PML and cmp files are compatible
        '''

        cfg = self.DFC.cfg
        test_temp_dir = "/scratch/tmp-mw545/voicebank_208_speakers/test_temp"

        total_file_count = 0
        diff_file_count  = 0
        diff_file_list   = []

        file_id_list = read_file_list(cfg.file_id_list_file['used'])
        for file_id in file_id_list:
            # 10% use
            x = numpy.random.randint(20)
            if x == 6:
                # File being used
                total_file_count += 1
                wav_file = os.path.join(cfg.wav_dir, file_id + '.wav')
                cmp_file_1 = os.path.join(cfg.nn_feat_dirs['cmp'], file_id + '.cmp')
                cmp_file_2 = os.path.join(test_temp_dir, file_id + '.cmp')
                vocoder_file_dict = {}
                for feat_name in cfg.acoustic_features:
                    vocoder_file_dict[feat_name] = os.path.join(test_temp_dir, file_id + cfg.acoustic_file_ext_dict[feat_name])
                self.DFC.wav_2_acoustic(wav_file, vocoder_file_dict)
                self.DFC.pml_2_cmp(vocoder_file_dict, cmp_file_2)

                cmp_data_1, frame_number_1 = self.DF_IO.load_data_file_frame(cmp_file_1, cfg.nn_feature_dims['cmp'])
                cmp_data_2, frame_number_2 = self.DF_IO.load_data_file_frame(cmp_file_2, cfg.nn_feature_dims['cmp'])

                if (frame_number_1 == frame_number_2) and (cmp_data_1 == cmp_data_2).all():
                    # remove temporary files
                    # self.logger.info('Removing file %s' % cmp_file_2)
                    os.remove(cmp_file_2)
                    for feat_name in cfg.acoustic_features:
                        os.remove(vocoder_file_dict[feat_name])
                else:
                    self.logger.info('Different file found! %s' % file_id)
                    diff_file_list.append(file_id)
                    diff_file_count += 1
                    if frame_number_1 == frame_number_2:
                        # Log number of values that are different
                        n = numpy.sum(cmp_data_1 != cmp_data_2)
                        self.logger.info('%i values are different' % n)
                        self.logger.info(cmp_data_1[cmp_data_1 != cmp_data_2])
                        self.logger.info(cmp_data_2[cmp_data_1 != cmp_data_2])
                        for feat_name in cfg.acoustic_features:
                            data_1 = self.DF_IO.read_pml_by_name_feat(file_id, feat_name)
                            file_name_2 = os.path.join(test_temp_dir, file_id + cfg.acoustic_file_ext_dict[feat_name])
                            data_2, frame_number = self.DF_IO.load_data_file_frame(file_name_2, cfg.acoustic_in_dimension_dict[feat_name])
                            if (data_1 == data_2).all():
                                pass
                            else:
                                n = numpy.sum(data_1 != data_2)
                                self.logger.info('%i values are different' % n)
                                self.logger.info(feat_name)
                    else:
                        self.logger.info('Different frame numbers: %s %s; %i %i' % (cmp_file_1,cmp_file_2, frame_number_1,frame_number_2))

        self.logger.info('Temp directory %s' % test_temp_dir)
        self.logger.info('cmp directory %s' % cfg.nn_feat_dirs['cmp'])
        self.logger.info('Different file numbers: %i out of %i' % (diff_file_count,total_file_count))
        print(diff_file_list)

    def cmp_data_test_v2(self):
        '''
        In previous test, it seems cmp files are more consistent than pml files
        i.e. in 10%  of the files, cmp files have <10 different values
        but in those files, pml files have a lot of different values, e.g. 70
        Therefore, the pml_2_cmp process should be questioned
        1. Generate cmp file
        2. Compare with current cmp file; log file name if not equal
        Remove cmp files, if equal

        Result:
        Inconsistent; maybe different versions of vocoder
        
        Action:
        Re-generating pml files
        '''

        cfg = self.DFC.cfg
        test_temp_dir = "/scratch/tmp-mw545/voicebank_208_speakers/test_temp"

        total_file_count = 0
        diff_file_count  = 0
        diff_file_list   = []

        file_id_list = read_file_list(cfg.file_id_list_file['used'])
        for file_id in file_id_list:
            # 10% use
            x = numpy.random.randint(20)
            if x == 6:
                # File being used
                total_file_count += 1
                cmp_file_1 = os.path.join(cfg.nn_feat_dirs['cmp'], file_id + '.cmp')
                cmp_file_2 = os.path.join(test_temp_dir, file_id + '.cmp')
                vocoder_file_dict = {}
                for feat_name in cfg.acoustic_features:
                    vocoder_file_dict[feat_name] = os.path.join(cfg.acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name])
                self.DFC.pml_2_cmp(vocoder_file_dict, cmp_file_2)

                cmp_data_1, frame_number_1 = self.DF_IO.load_data_file_frame(cmp_file_1, cfg.nn_feature_dims['cmp'])
                cmp_data_2, frame_number_2 = self.DF_IO.load_data_file_frame(cmp_file_2, cfg.nn_feature_dims['cmp'])

                if (frame_number_1 == frame_number_2) and (cmp_data_1 == cmp_data_2).all():
                    # remove temporary files
                    # self.logger.info('Removing file %s' % cmp_file_2)
                    os.remove(cmp_file_2)

                else:
                    self.logger.info('Different file found! %s' % file_id)
                    diff_file_list.append(file_id)
                    diff_file_count += 1
                    if frame_number_1 == frame_number_2:
                        # Log number of values that are different
                        n = numpy.sum(cmp_data_1 != cmp_data_2)
                        self.logger.info('%i values are different' % n)
                        # self.logger.info(cmp_data_1[cmp_data_1 != cmp_data_2])
                        # self.logger.info(cmp_data_2[cmp_data_1 != cmp_data_2])
                    else:
                        self.logger.info('Different frame numbers: %s %s; %i %i' % (cmp_file_1,cmp_file_2, frame_number_1,frame_number_2))
                    os.remove(cmp_file_2)

        self.logger.info('Temp directory %s' % test_temp_dir)
        self.logger.info('cmp directory %s' % cfg.nn_feat_dirs['cmp'])
        self.logger.info('Different file numbers: %i out of %i' % (diff_file_count,total_file_count))
        print(diff_file_list)

    def pml_2_wav_diff_pml_test(self, log_file_name=True):
        """ 
        Use old and new pml features to reconstruct wav file; also copy original wav file 

        Result:
        No noticeable difference, despite very different values
        """
        cfg = self.DFC.cfg
        file_id = 'p103_302'
        test_temp_dir = "/scratch/tmp-mw545/voicebank_208_speakers/test_temp"

        # check which feature has the error

        for feat_name in cfg.acoustic_features:
            data_1 = self.DF_IO.read_pml_by_name_feat(file_id, feat_name)
            file_name_2 = os.path.join(test_temp_dir, file_id + cfg.acoustic_file_ext_dict[feat_name])
            data_2, frame_number = self.DF_IO.load_data_file_frame(file_name_2, cfg.acoustic_in_dimension_dict[feat_name])
            if (data_1 == data_2).all():
                pass
            else:
                n = numpy.sum(data_1 != data_2)
                self.logger.info('%i values are different' % n)
                self.logger.info(feat_name)
                self.logger.info(file_name_2)

        wav_file = os.path.join(cfg.wav_dir, file_id + '.wav')
        wav_file_0 = os.path.join(test_temp_dir, file_id + '_0.wav')
        wav_file_1 = os.path.join(test_temp_dir, file_id + '_1.wav')
        wav_file_2 = os.path.join(test_temp_dir, file_id + '_2.wav')
        vocoder_file_dict_1 = {}
        vocoder_file_dict_2 = {}
        for feat_name in cfg.acoustic_features:
            vocoder_file_dict_1[feat_name] = os.path.join(cfg.acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name])
            vocoder_file_dict_2[feat_name] = os.path.join(test_temp_dir, file_id + cfg.acoustic_file_ext_dict[feat_name])

        self.DFC.wav_2_acoustic(wav_file, vocoder_file_dict_2)
        shutil.copyfile(wav_file, wav_file_0)
        self.DFC.acoustic_2_wav(vocoder_file_dict_1, wav_file_1)
        self.DFC.acoustic_2_wav(vocoder_file_dict_2, wav_file_2)

        if log_file_name:
            self.logger.info(wav_file_0)
            self.logger.info(wav_file_1)
            self.logger.info(wav_file_2)

    def lf0_16kHz_test(self):
        '''
        Test lf0_2_lf0_16kHz function
        Read lf0 file and wav_cmp file
        Generate lf0_16kHz data
        '''

        cfg = self.DFC.cfg
        # test_temp_dir = "/scratch/tmp-mw545/voicebank_208_speakers/test_temp"
        lf0_dir = '/home/dawna/tts/data/VoiceBank16kHz/PML/lf0'
        test_temp_dir = "/home/dawna/tts/mw545/Export_Temp/voicebank_208_speakers/lf0_16kHz_test"
        if not os.path.exists(test_temp_dir): os.makedirs(test_temp_dir)
        file_id_list = ['p290_028']

        for file_id in file_id_list:
            lf0_file = os.path.join(lf0_dir, file_id + '.lf0')
            lf016k_file   = os.path.join(test_temp_dir, file_id + '.lf016k')

            self.DFC.lf0_2_lf0_16kHz(lf0_file, lf016k_file)

            lf0_data, num_samples = self.DF_IO.load_data_file_frame(lf0_file, 1)
            lf016k_data, num_samples = self.DF_IO.load_data_file_frame(lf016k_file, 1)
            print(lf016k_file)
            print(lf0_data)
            print(lf016k_data)

class Data_Silence_Reducer_Test(object):
    """docstring for Data_Silence_Reducer_Test"""
    def __init__(self, cfg=None):
        super(Data_Silence_Reducer_Test, self).__init__()
        self.cfg = cfg
        self.logger = make_logger("DSR_Test")

        self.cfg = cfg
        self.DF_IO = Data_File_IO(cfg)
        self.graph_plotter = Graph_Plotting()
        self.DRT = Data_Replicate_Test(cfg)

        from frontend_mw545.data_silence_reducer import Data_Silence_Reducer
        self.DSR = Data_Silence_Reducer(cfg)

    def silence_reducer_test(self):
        '''
        Check if the current nn_cmp_resil is consistent with a new generated file
        '''
        cfg = self.cfg
        file_id_list = ['p290_028']

        output_dir = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/temp_test_dir'
        feat_name = 'pitch'
        
        for file_id in file_id_list:
            l = os.path.join(self.cfg.lab_dir, file_id + '.lab')
            x = os.path.join(self.cfg.nn_feat_dirs[feat_name], file_id + '.'+feat_name)
            y = os.path.join(output_dir, file_id + '.'+feat_name)
            r = os.path.join(self.cfg.nn_feat_resil_dirs[feat_name], file_id + '.'+feat_name)

            self.DSR.reduce_silence_file(l, x, y, feat_name=feat_name)

            if self.DRT.check_file_same(y, r):
                self.logger.info('Matched!')
            else:
                # Mismatch found, print files names
                self.logger.info(y)
                self.logger.info(r)

    def amount_of_silence_test(self):
        '''
        Calculate number of silence frames before and after, for each file
        Keep all files <= 10
        '''
        file_name = os.path.join('/home/dawna/tts/mw545/TorchDV/file_id_lists/data_meta', 'file_id_list_num_sil_frame.scp')

        file_id_list = []

        sil_min_all = 1000

        fid = open(file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            x_list = line.split(' ')

            file_id = x_list[0]
            l = int(x_list[1])
            x = int(x_list[2])
            y = int(x_list[3])
            sil_start = x
            sil_end   = l - y - 1
            sil_min = min(sil_start, sil_end)
            if sil_min <= 5:
                file_line = '%s %i %i' %(file_id, sil_start, sil_end)
                file_id_list.append(file_line)
                self.logger.info(file_line)
                if sil_min < sil_min_all:
                    sil_min_all = sil_min
        fid.close()
        self.logger.info('Min of all is %i' % sil_min_all)

class Data_Wav_Min_Max_Normaliser_test(object):
    """docstring for Data_Wav_Min_Max_Normaliser_test"""
    def __init__(self, cfg=None):
        super(Data_Wav_Min_Max_Normaliser_test, self).__init__()
        self.cfg = cfg
        self.logger = make_logger("WavNorm_Test")

        self.cfg = cfg
        self.DF_IO = Data_File_IO(cfg)

        from frontend_mw545.data_norm import Data_Wav_Min_Max_Normaliser
        self.DWMMN = Data_Wav_Min_Max_Normaliser(cfg)

        self.DRT = Data_Replicate_Test(cfg)

    def normaliser_test(self):
        cfg = self.cfg
        # file_id_list = ['p290_028']

        file_id_list_file = self.cfg.file_id_list_file['used']
        self.logger.info('Reading file list from %s' % file_id_list_file)
        file_id_list = read_file_list(file_id_list_file)

        # Draw a few files for testing
        list_draw = numpy.random.choice(file_id_list, 100, replace=False)
        file_id_list = list_draw

        output_dir = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/temp_test_dir'
        feat_name = 'wav'

        for file_id in file_id_list:
            x = os.path.join(self.cfg.nn_feat_resil_dirs[feat_name], file_id + '.'+feat_name)
            y = os.path.join(output_dir, file_id + '.'+feat_name)
            r = os.path.join(self.cfg.nn_feat_resil_norm_dirs[feat_name], file_id + '.'+feat_name)
            x_2 = y + '.recon'

            self.DWMMN.norm_file(x, y)
            self.DWMMN.denorm_file(y, x_2)

            # if self.DRT.check_file_same(y, r):
            #     self.logger.info('Matched!')
            # else:
            #     # Mismatch found, print files names
            #     self.logger.info(y)
            #     self.logger.info(r)

            if self.DRT.check_file_same(x, x_2):
                self.logger.info('Matched!')
            else:
                # Mismatch found, print files names
                self.logger.info(x)
                self.logger.info(x_2)

class Data_CMP_Mean_Var_Normaliser_test(object):
    """docstring for Data_Wav_Min_Max_Normaliser_test"""
    def __init__(self, cfg=None):
        super(Data_CMP_Mean_Var_Normaliser_test, self).__init__()
        self.cfg = cfg
        self.logger = make_logger("CMPNorm_Test")

        self.cfg = cfg
        self.DF_IO = Data_File_IO(cfg)

        from frontend_mw545.data_norm import Data_Mean_Var_Normaliser
        self.DMVN = Data_Mean_Var_Normaliser(cfg)

        self.DRT = Data_Replicate_Test(cfg)

    def test(self):
        # self.norm_info_file_load_test()
        # self.normaliser_test()
        # self.make_file_id_list()
        self.compute_normaliser_test_1()

    def norm_info_file_load_test(self):
        self.DMVN.load_mean_std_values()
        print(self.DMVN.mean_vector)
        print(self.DMVN.std_vector)
        self.DMVN.save_mean_std_values(norm_info_file='/home/dawna/tts/mw545/TorchDV/debug_nausicaa/temp_test_dir/cmp_info.dat')

    def normaliser_test(self):
        cfg = self.cfg
        # file_id_list = ['p290_028']

        file_id_list_file = self.cfg.file_id_list_file['used']
        self.logger.info('Reading file list from %s' % file_id_list_file)
        file_id_list = read_file_list(file_id_list_file)

        # Draw a few files for testing
        list_draw = numpy.random.choice(file_id_list, 100, replace=False)
        file_id_list = list_draw

        output_dir = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/temp_test_dir'
        feat_name = 'cmp'
        self.DMVN.load_mean_std_values(cfg.nn_feat_resil_norm_files['cmp'])

        for file_id in file_id_list:
            x = os.path.join(self.cfg.nn_feat_resil_dirs[feat_name], file_id + '.'+feat_name)
            y = os.path.join(output_dir, file_id + '.'+feat_name)
            r = os.path.join(self.cfg.nn_feat_resil_norm_dirs[feat_name], file_id + '.'+feat_name)
            x_2 = y + '.recon'

            self.DMVN.norm_file(x, y)
            # self.DWMMN.denorm_file(y, x_2)

            if not self.DRT.check_file_same(r, y):
                self.logger.info('Different files %s %s' %(r, y))
            # else:
            #     # Mismatch found, print files names
            #     self.logger.info(y)
            #     self.logger.info(r)

            # if self.DRT.check_file_same(x, x_2):
            #     self.logger.info('Matched!')
            # else:
            #     # Mismatch found, print files names
            #     self.logger.info(x)
            #     self.logger.info(x_2)

    def compute_normaliser_test_1(self):
        '''
        Result: Good
            Almost the same as current norm_info_file, precision difference
            All files stored float32 for storage reduction; freshly computed values are float64
            Time: 
        '''
        # file_id_list = self.make_file_id_list()
        self.logger.info('start computing mean')
        self.DMVN.compute_mean()
        self.logger.info('start computing std')
        self.DMVN.compute_std()

class Data_Loader_Test(object):
    """docstring for Data_Loader_Test"""
    def __init__(self, cfg=None):
        super(Data_Loader_Test, self).__init__()
        self.cfg = cfg
        self.logger = make_logger("Data_Loader_Test")

        self.cfg = cfg
        self.DF_IO = Data_File_IO(cfg)

        from exp_mw545.exp_dv_config import dv_y_configuration
        self.dv_y_cfg = dv_y_configuration(cfg)
        self.dv_y_cfg.auto_complete()

        self.DRT = Data_Replicate_Test()

    def single_file_data_loader_test(self):
        '''
        Compare 2 methods of loading single file
        One is ref, slow but stable method
        Result: Consistent, and correct
        '''
        from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Ref
        from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Single_File
        ref_data_loader = Build_dv_y_wav_data_loader_Ref(cfg=self.cfg, dv_y_cfg=self.dv_y_cfg)
        new_data_loader = Build_dv_y_wav_data_loader_Single_File(cfg=self.cfg, dv_y_cfg=self.dv_y_cfg)

        # from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Single_File
        # from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Single_File_v2
        # ref_data_loader = Build_dv_y_wav_data_loader_Single_File(cfg=self.cfg, dv_y_cfg=self.dv_y_cfg)
        # new_data_loader = Build_dv_y_wav_data_loader_Single_File_v2(cfg=self.cfg, dv_y_cfg=self.dv_y_cfg)

        file_id_list_file = self.cfg.file_id_list_file['dv_enough']
        self.logger.info('Reading file list from %s' % file_id_list_file)
        file_id_list = read_file_list(file_id_list_file)

        # Draw a few files for testing
        list_draw = numpy.random.choice(file_id_list, 100, replace=False)
        file_id_list = list_draw

        # file_id_list = ['p290_028'] # p290_028 561 115 489; first pitch 0.628688
        for file_id in file_id_list:
            for start_sample_no_sil in range(100):
                self.logger.info('start_sample_no_sil is %i' % start_sample_no_sil)
                ref_feed_dict = ref_data_loader.make_feed_dict(file_id, start_sample_no_sil)
                new_feed_dict = new_data_loader.make_feed_dict(file_id, start_sample_no_sil)

                self.DRT.check_data_dict_same(ref_feed_dict, new_feed_dict, tol=1./100000000)

                # for k in ref_feed_dict:
                #     print(k)
                #     print(ref_feed_dict[k][0])
                #     print(new_feed_dict[k][0])

    def multi_files_data_loader_test_1(self):
        '''
        This test compares the output when start_sample_no_sil_list is provided
        Test 2 will test if it is None
        Result: Consistent
        '''

        from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Single_File
        from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Multi_Speaker

        single_data_loader = Build_dv_y_wav_data_loader_Single_File(cfg=self.cfg, dv_y_cfg=self.dv_y_cfg)
        multi_data_loader  = Build_dv_y_wav_data_loader_Multi_Speaker(cfg=self.cfg, dv_y_cfg=self.dv_y_cfg)

        file_id_list_file = self.cfg.file_id_list_file['used']
        self.logger.info('Reading file list from %s' % file_id_list_file)
        file_id_list = read_file_list(file_id_list_file)

        # Draw a few files for testing
        list_draw = numpy.random.choice(file_id_list, self.dv_y_cfg.batch_num_spk, replace=False)
        file_id_list = list_draw

        start_sample_no_sil_list = range(self.dv_y_cfg.batch_num_spk)

        multi_feed_dict = multi_data_loader.make_feed_dict(file_id_list, start_sample_no_sil_list)

        for i, file_id in enumerate(file_id_list):
            start_sample_no_sil = start_sample_no_sil_list[i]
            single_feed_dict = single_data_loader.make_feed_dict(file_id, start_sample_no_sil)

            self.DRT.check_data_same(multi_feed_dict['wav_ST'][i], single_feed_dict['wav_T'])
            self.DRT.check_data_same(multi_feed_dict['tau_SBM'][i], single_feed_dict['tau_BM'])
            self.DRT.check_data_same(multi_feed_dict['vuv_SBM'][i], single_feed_dict['vuv_BM'])
            self.DRT.check_data_same(multi_feed_dict['f_SBM'][i], single_feed_dict['f_BM'])

    def multi_files_data_loader_test_2(self):
        '''
        This test compares the output when start_sample_no_sil_list is None
        Result: no error encountered
        '''
        from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Multi_Speaker

        multi_data_loader  = Build_dv_y_wav_data_loader_Multi_Speaker(cfg=self.cfg, dv_y_cfg=self.dv_y_cfg)

        file_id_list_file = self.cfg.file_id_list_file['used']
        self.logger.info('Reading file list from %s' % file_id_list_file)
        file_id_list = read_file_list(file_id_list_file)

        # Draw a few files for testing
        for i in range(10000):
            list_draw = numpy.random.choice(file_id_list, self.dv_y_cfg.batch_num_spk, replace=False)
            file_id_list = list_draw

            multi_feed_dict = multi_data_loader.make_feed_dict(file_id_list)

    def dv_y_train_data_loader_test_1(self):
        '''
        Test if it is functional?
        '''
        from frontend_mw545.data_loader import Build_dv_y_train_data_loader
        from frontend_mw545.data_loader import Build_dv_y_wav_data_loader_Single_File

        dv_y_train_data_loader = Build_dv_y_train_data_loader(cfg=self.cfg, dv_y_cfg=self.dv_y_cfg)
        single_data_loader = Build_dv_y_wav_data_loader_Single_File(cfg=self.cfg, dv_y_cfg=self.dv_y_cfg)

        start_sample_no_sil_list = range(self.dv_y_cfg.batch_num_spk)
        # file_id_list = dv_y_train_data_loader.draw_n_files(utter_tvt_name='train')
        train_feed_dict, batch_size = dv_y_train_data_loader.make_feed_dict(utter_tvt_name='train', start_sample_no_sil_list=None)
        print(train_feed_dict.keys())
        for k in train_feed_dict.keys():
            print(k)
            print(train_feed_dict[k].shape)
        print(train_feed_dict['one_hot'].tolist())

    def dv_y_train_data_loader_test_2(self):
        '''
        Test a large number of batches
        '''
        from frontend_mw545.data_loader import Build_dv_y_train_data_loader
        dv_y_train_data_loader = Build_dv_y_train_data_loader(cfg=self.cfg, dv_y_cfg=self.dv_y_cfg)
        for i in range(100000):
            feed_dict, batch_size = dv_y_train_data_loader.make_feed_dict(utter_tvt_name='train')
        # print(file_id_list)

def numpy_random_speed_test():
    '''
    Result: better to draw all ratios then multiply, than draw one integer at a time
    '''
    test_start_time = time.time()
    for i in range(1000):
        for j in range(1000):
            x = numpy.random.randint(0, 101) 
    test_end_time = time.time()
    print('%s' % str(test_end_time-test_start_time))

    test_start_time = time.time()
    for i in range(1000):
        r_list = numpy.random.rand(1000)
        for j in range(1000):
            x = int(r_list[j] * 100 + 0.5)
    test_end_time = time.time()
    print('%s' % str(test_end_time-test_start_time))

def shortest_files(cfg):
    from frontend_mw545.data_io import Data_Meta_List_File_IO
    DMLF_IO = Data_Meta_List_File_IO(cfg)
    meta_file_name = '/home/dawna/tts/mw545/TorchDV/file_id_lists/data_meta/file_id_list_num_sil_frame.scp'

    file_frame_dict = DMLF_IO.read_file_list_num_silence_frame(meta_file_name)

    min_file_len = 400
    file_counter = 0

    for file_id in file_frame_dict:
        l, x, y = file_frame_dict[file_id]
        file_len = y - x + 1

        if file_len < min_file_len:
            print('%s %s' %(file_id, file_len))
            file_counter += 1
    print('Total %i files shorter than %i frames' %(file_counter, min_file_len))

def longest_files(cfg):
    from frontend_mw545.data_io import Data_Meta_List_File_IO
    DMLF_IO = Data_Meta_List_File_IO(cfg)
    meta_file_name = '/home/dawna/tts/mw545/TorchDV/file_id_lists/data_meta/file_id_list_num_sil_frame.scp'

    file_frame_dict = DMLF_IO.read_file_list_num_silence_frame(meta_file_name)

    min_file_len = 2300
    file_counter = 0

    for file_id in file_frame_dict:
        l, x, y = file_frame_dict[file_id]
        file_len = y - x + 1

        if file_len > min_file_len:
            print('%s %s' %(file_id, file_len))
            file_counter += 1

    print('Total %i files longer than %i frames' %(file_counter, min_file_len))

def dv_file_list_len_test(cfg):
    from exp_mw545.exp_dv_config import dv_y_configuration
    dv_y_cfg = dv_y_configuration(cfg)

    from frontend_mw545.data_loader import Build_dv_selecter
    dv_selecter = Build_dv_selecter(cfg, dv_y_cfg)
    # for k in dv_selecter.file_list_dict:
    #     print(k)
    #     print(len(dv_selecter.file_list_dict[k]))

    len_list = {}
    for utter_tvt_name in ['train', 'valid', 'test']:
        len_list[utter_tvt_name] = []
                
    for speaker_id in dv_selecter.speaker_id_list:
        for utter_tvt_name in ['train', 'valid', 'test']:
            len_list[utter_tvt_name].append(len(dv_selecter.file_list_dict[(speaker_id, utter_tvt_name)]))

    for utter_tvt_name in ['train', 'valid', 'test']:
        print(len_list[utter_tvt_name])
        print(max(len_list[utter_tvt_name]))
        print(min(len_list[utter_tvt_name]))


def temp_test(cfg):
    # average the last dimension
    n = 4
    a = numpy.random.rand(1,2,7)
    
    n_last = a.shape[-1]
    K = int(n_last / n)
    n_res = n_last - K * n
    
    b = numpy.zeros(a.shape)
    d = a.ndim

    # Deal with K*n first
    for k in range(K):
        index_list = range(k*n, (k+1)*n)
        for i in index_list:
            if d == 3:
                b[:,:,i] = numpy.mean(a[:,:,index_list], -1)

    # Deal with residuals
    if n_res > 0:
        index_list = range(K*n, n_last)
        for i in index_list:
            if d == 3:
                b[:,:,i] = numpy.mean(a[:,:,index_list], -1)

    print(a)
    print(b)





class Data_Vocoder_Position_Test(object):
    """docstring for Data_Vocoder_Position_Test"""
    def __init__(self, cfg=None):
        super(Data_Vocoder_Position_Test, self).__init__()
        self.logger = make_logger("Voc_Pos_Test")
        self.cfg = cfg

        self.test_file_list = read_file_list(cfg.file_id_list_file['dv_pos_test']) # 186^5 files
        self.output_dir = '/data/mifs_scratch/mjfg/mw545/dv_pos_test'

    def test(self):
        self.vocoder_test_2()

    def run(self):
        self.pml_2_cmp_norm()

    def check_file_len(self, file_id):
        '''
        Return the length of a file
        Number of non-silence frames
        '''
        try:
            self.file_frame_dict
        except:
            from frontend_mw545.data_io import Data_Meta_List_File_IO
            DMLF_IO = Data_Meta_List_File_IO()

            meta_file_name = '/home/dawna/tts/mw545/TorchDV/file_id_lists/data_meta/file_id_list_num_sil_frame.scp'
            self.file_frame_dict = DMLF_IO.read_file_list_num_silence_frame(meta_file_name)

        l, x, y = self.file_frame_dict[file_id]
        file_len = y - x + 1

        return file_len

    def check_file_list(self):
        '''
        Check the file length in dv_test
        Result: this file_id_list contains files shorter than 1s
            Those files are not even in file_id_list_used_cfg_dv_enough.scp
            Thus need to write a new file_id_list, contains only files long enough
        '''
        cfg = self.cfg
        in_file_list_name = cfg.file_id_list_file['dv_test']
        in_file_list = self.DLFIO.read_file_list(in_file_list_name)
        

        min_file_len = 250
        file_counter = 0

        for file_id in in_file_list:
            file_len = self.check_file_len(file_id)

            if file_len < min_file_len:
                print('%s %s' %(file_id, file_len))
                file_counter += 1
        print('Total %i files shorter than %i frames' %(file_counter, min_file_len))

    def make_file_list(self):
        '''
        Select 5 files from each test speaker
        Select files long enough for dv and for positional test, 210 frames
        Make and store a new file list
        Result: Done, good, 186*5 files in the file_list
        '''
        cfg = self.cfg
        min_file_len = 210
        in_file_list_name = cfg.file_id_list_file['dv_test']
        out_file_list_name = cfg.file_id_list_file['dv_pos_test']

        in_file_list = self.DLFIO.read_file_list(in_file_list_name)
        # Split into speakers; draw 5 each
        speaker_id_list = cfg.speaker_id_list_dict['train']
        from frontend_mw545.modules import File_List_Selecter
        FL_Selecter = File_List_Selecter()
        in_file_dict = FL_Selecter.sort_by_speaker_list(in_file_list, speaker_id_list)
        out_file_list = []
        for speaker_id in speaker_id_list:
            file_counter = 0
            for file_id in in_file_dict[speaker_id]:
                file_len = self.check_file_len(file_id)
                if file_len < min_file_len:
                    pass
                else:
                    out_file_list.append(file_id)
                    file_counter += 1
                    if file_counter == 5:
                        break
        self.DLFIO.write_file_list(out_file_list, out_file_list_name)

    def vocoder_test_1(self):
        '''
        Check with current vocoder features
        Result: consistent
        '''
        cfg = self.cfg
        DRT = Data_Replicate_Test()
        from frontend_mw545.data_converter import Data_File_Converter
        DFC = Data_File_Converter(cfg)

        file_list = numpy.random.choice(self.test_file_list, 10, replace=False)

        new_acoustic_data_dir = os.path.join(self.output_dir, 'PML')
        new_acoustic_dir_dict = {}
        for feat_name in cfg.acoustic_features:
            new_acoustic_dir_dict[feat_name] = os.path.join(new_acoustic_data_dir, feat_name)
            prepare_script_file_path(new_acoustic_dir_dict[feat_name])

        ref_vocoder_file_dict = {}
        new_vocoder_file_dict = {}
        for file_id in file_list:
            wav_file = os.path.join(cfg.wav_dir, file_id + '.wav')
            for feat_name in cfg.acoustic_features:
                ref_vocoder_file_dict[feat_name] = os.path.join(cfg.acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name])
                new_vocoder_file_dict[feat_name] = os.path.join(new_acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name])
            self.logger.info('Generating for file %s' % wav_file)
            DFC.wav_2_acoustic(wav_file, new_vocoder_file_dict)
            DRT.check_file_dict_same(ref_vocoder_file_dict, new_vocoder_file_dict)

    def vocoder_test_2(self):
        '''
        Check that after removing 80 samples, vocoder features are same but shifted
            wav --> data; remove 80; data --> wav; new_wav --> PML
            orig_wav --> PML, remove 1 frame; compare
        Result:
            1. file length is NOT -1; some are -1, some -3, some even +1
            2. PML feature NOT equal, after shift 1 frame
            3. Conclusion: shift 1 frame is not a viable idea
        '''
        cfg = self.cfg
        DRT = Data_Replicate_Test()

        from frontend_mw545.data_io import Data_File_IO
        DF_IO = Data_File_IO(cfg)
        from frontend_mw545.data_converter import Data_File_Converter
        DFC = Data_File_Converter(cfg)

        file_list = numpy.random.choice(self.test_file_list, 10, replace=False)

        new_wav_data_dir = os.path.join(self.output_dir, 'wav_shift')
        wav_cmp_data_dir = os.path.join(self.output_dir, 'wav_cmp')
        prepare_script_file_path(new_wav_data_dir)
        prepare_script_file_path(wav_cmp_data_dir)

        new_acoustic_data_dir = os.path.join(self.output_dir, 'PML_shift')
        new_acoustic_dir_dict = {}
        ref_acoustic_data_dir = os.path.join(self.output_dir, 'PML')
        ref_acoustic_dir_dict = {}
        for feat_name in cfg.acoustic_features:
            new_acoustic_dir_dict[feat_name] = os.path.join(new_acoustic_data_dir, feat_name)
            ref_acoustic_dir_dict[feat_name] = os.path.join(ref_acoustic_data_dir, feat_name)
            prepare_script_file_path(new_acoustic_dir_dict[feat_name])
            prepare_script_file_path(ref_acoustic_dir_dict[feat_name])

        ref_vocoder_file_dict = {}
        new_vocoder_file_dict = {}
        for file_id in file_list:
            wav_file = os.path.join(cfg.wav_dir, file_id + '.wav')
            for feat_name in cfg.acoustic_features:
                ref_vocoder_file_dict[feat_name] = os.path.join(ref_acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name])
                new_vocoder_file_dict[feat_name] = os.path.join(new_acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name])
            # wav --> data; remove 80; data --> wav; new_wav --> PML
            wav_data = DF_IO.read_wav_2_wav_1D_data(in_file_name=wav_file)
            wav_data_new = wav_data[80:]
            new_wav_file = os.path.join(new_wav_data_dir, file_id + '.wav.80')
            DF_IO.write_wav_1D_data_2_wav(wav_data_new, new_wav_file)
            DFC.wav_2_acoustic(new_wav_file, new_vocoder_file_dict)

            # orig_wav --> PML, remove 1 frame; compare
            DFC.wav_2_acoustic(wav_file, ref_vocoder_file_dict)
            for feat_name in cfg.acoustic_features:
                ref_data, ref_l = DF_IO.load_data_file_frame((ref_vocoder_file_dict[feat_name]), cfg.acoustic_in_dimension_dict[feat_name], return_frame_number=True)
                new_data, new_l = DF_IO.load_data_file_frame((new_vocoder_file_dict[feat_name]), cfg.acoustic_in_dimension_dict[feat_name], return_frame_number=True)
                l = 40
                DRT.check_data_same(ref_data[1:41], new_data[0:40])

    def vocoder_run(self):
        '''
        Generate for all files, up to 50 samples shift
        wav --> data; remove 80; data --> wav; new_wav --> PML
        '''

        cfg = self.cfg

        from frontend_mw545.data_io import Data_File_IO
        DF_IO = Data_File_IO(cfg)
        from frontend_mw545.data_converter import Data_File_Converter
        DFC = Data_File_Converter(cfg)
        from frontend_mw545.modules import prepare_script_file_path

        # Run separately, split into 5 groups
        # ~7s per file; 930*51*7=332010s=92h
        file_list = self.test_file_list[930/5*4:930/5*5]

        new_wav_data_dir = os.path.join(self.output_dir, 'wav_shift')
        prepare_script_file_path(new_wav_data_dir)

        new_acoustic_data_dir = os.path.join(self.output_dir, 'PML_shift')
        new_acoustic_dir_dict = {}
        for feat_name in cfg.acoustic_features:
            new_acoustic_dir_dict[feat_name] = os.path.join(new_acoustic_data_dir, feat_name)
            prepare_script_file_path(new_acoustic_dir_dict[feat_name])

        new_vocoder_file_dict = {}
        for file_id in file_list:
            wav_file = os.path.join(cfg.wav_dir, file_id + '.wav')
            wav_data = DF_IO.read_wav_2_wav_1D_data(in_file_name=wav_file)
            for i in range(51):
                wav_data_new = wav_data[i:]
                new_wav_file = os.path.join(new_wav_data_dir, file_id + '.wav.%i'%i)
                DF_IO.write_wav_1D_data_2_wav(wav_data_new, new_wav_file)
                for feat_name in cfg.acoustic_features:
                    new_vocoder_file_dict[feat_name] = os.path.join(new_acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name]+'.%i'%i)
                self.logger.info('Generating for file %s' % new_wav_file)
                DFC.wav_2_acoustic(new_wav_file, new_vocoder_file_dict)

    def pml_2_cmp_norm(self):
        '''
        1. pml -> cmp
        2. cmp -> cmp_resil
        3. cmp_resil -> cmp_resil_norm
        '''
        cfg = self.cfg

        file_list = self.test_file_list

        from frontend_mw545.data_converter import Data_File_Converter
        DFC = Data_File_Converter(cfg)
        from frontend_mw545.data_silence_reducer import Data_Silence_Reducer
        DSR = Data_Silence_Reducer(cfg)
        from frontend_mw545.data_norm import Data_Mean_Var_Normaliser
        DMVN = Data_Mean_Var_Normaliser(cfg)
        norm_info_file = cfg.nn_feat_resil_norm_files['cmp']
        DMVN.load_mean_std_values(norm_info_file)

        new_acoustic_data_dir = os.path.join(self.output_dir, 'PML_shift')
        new_acoustic_dir_dict = {}
        for feat_name in cfg.acoustic_features:
            new_acoustic_dir_dict[feat_name] = os.path.join(new_acoustic_data_dir, feat_name)
            prepare_script_file_path(new_acoustic_dir_dict[feat_name])

        temp_dir = os.path.join(self.output_dir, 'temp')
        output_dir = os.path.join(self.output_dir, 'cmp_shift')
        prepare_script_file_path(temp_dir)
        prepare_script_file_path(output_dir)

        new_vocoder_file_dict = {}
        for file_id in file_list:
            alignment_file_name = os.path.join(cfg.lab_dir, file_id + '.lab')
            for i in range(51):
                for feat_name in cfg.acoustic_features:
                    new_vocoder_file_dict[feat_name] = os.path.join(new_acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name]+'.%i'%i)         
                cmp_file = os.path.join(temp_dir, file_id + '.cmp.%i'%i)
                cmp_resil_file = os.path.join(temp_dir, file_id + '.cmp_resil.%i'%i)
                cmp_resil_norm_file = os.path.join(output_dir, file_id + '.cmp.%i'%i)

                DFC.pml_2_cmp(new_vocoder_file_dict, cmp_file)
                DSR.reduce_silence_file(alignment_file_name, cmp_file, cmp_resil_file, feat_name='cmp')
                DMVN.norm_file(cmp_resil_file, cmp_resil_norm_file, feat_name='cmp')




##########################
# Data Tests per Feature #
##########################
  
class Data_Test_Base(object):
    """docstring for Data_Test_Base"""
    def __init__(self, cfg=None):
        super(Data_Test_Base, self).__init__()
        self.cfg = cfg

        self.DF_IO = Data_File_IO(cfg)
        self.DRT = Data_Replicate_Test(cfg)
        self.file_id_list = read_file_list(cfg.file_id_list_file['used'])

        self.temp_dir_root = '/scratch/tmp-mw545/voicebank_208_speakers/test_temp'
        self.temp_dir = self.make_temp_dir()

    def make_temp_dir(self):
        '''
        Make a temporary directory for each test
        Clean/Remove after each test
        '''
        temp_dir_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        temp_dir = os.path.join(self.temp_dir_root, temp_dir_name)
        self.logger.info('Temp directory is %s ' % temp_dir)
        prepare_script_file_path(temp_dir)
        return temp_dir
        
    def clear_temp_file(self, file_name):
        '''
        Remove the temporary file generated
        '''
        os.remove(file_name)

    def clear_temp_dir(self, temp_dir=None):
        '''
        Remove the entire temp_dir
        '''
        if temp_dir is None:
            temp_dir = self.temp_dir
        shutil.rmtree(temp_dir)


class Data_Acoustic_Pipeline_Test(Data_Test_Base):
    """docstring for Data_Vocoder_Pipeline_Test"""
    def __init__(self, cfg=None):
        self.logger = make_logger("Acoustic_Test")
        super(Data_Acoustic_Pipeline_Test, self).__init__(cfg)

    def test(self):
        self.cmp_norm_test()
        self.cmp_denorm_test()
        self.clear_temp_dir()

    def wav_2_pml_test(self):
        '''
        wav --> pml
        Compare pml vs current pml
        Result: Good for 1000 files
        '''
        cfg = self.cfg
        file_list = self.file_id_list[:1000]

        from frontend_mw545.data_converter import Data_File_Converter
        DFC = Data_File_Converter(cfg)

        new_acoustic_dir_dict = {}
        for feat_name in cfg.acoustic_features:
            new_acoustic_dir_dict[feat_name] = os.path.join(self.temp_dir, feat_name)
            prepare_script_file_path(new_acoustic_dir_dict[feat_name])

        ref_vocoder_file_dict = {}
        new_vocoder_file_dict = {}
        for file_id in file_list:
            wav_file = os.path.join(cfg.wav_dir, file_id + '.wav')
            for feat_name in cfg.acoustic_features:
                ref_vocoder_file_dict[feat_name] = os.path.join(cfg.acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name])
                new_vocoder_file_dict[feat_name] = os.path.join(new_acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name])

            DFC.wav_2_acoustic(wav_file, new_vocoder_file_dict)
            if not self.DRT.check_file_dict_same(ref_vocoder_file_dict, new_vocoder_file_dict):
                self.logger.info('Difference when generating for file %s' % wav_file)

    def pml_2_cmp_test(self):
        '''
        pml --> cmp
        Compare cmp vs current cmp
        Results: Good, 10000 files
        '''
        cfg = self.cfg
        file_list = self.file_id_list[:10000]

        from frontend_mw545.data_converter import Data_File_Converter
        DFC = Data_File_Converter(cfg)

        ref_vocoder_file_dict = {}
        for file_id in file_list:
            ref_cmp_file = os.path.join(cfg.nn_feat_dirs['cmp'], file_id + '.cmp')
            new_cmp_file = os.path.join(self.temp_dir, file_id + '.cmp')
            for feat_name in cfg.acoustic_features:
                ref_vocoder_file_dict[feat_name] = os.path.join(cfg.acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name])

            DFC.pml_2_cmp(ref_vocoder_file_dict, new_cmp_file)
            if not self.DRT.check_file_same(ref_cmp_file, new_cmp_file):
                self.logger.info('Difference when generating to file %s' % new_cmp_file)

    def cmp_resil_test(self):
        '''
        cmp --> cmp_resil
        Compare cmp_resil vs current cmp_resil
        Results: Good, 10000 files
            Need to pad end; beginning has sufficient silence
        '''
        cfg = self.cfg
        file_list = self.file_id_list[:10000]

        from frontend_mw545.data_silence_reducer import Data_Silence_Reducer
        DSR = Data_Silence_Reducer(cfg)

        for file_id in file_list:
            alignment_file_name = os.path.join(cfg.lab_dir, file_id + '.lab')
            ori_cmp_file = os.path.join(cfg.nn_feat_dirs['cmp'], file_id + '.cmp')
            ref_cmp_file = os.path.join(cfg.nn_feat_resil_dirs['cmp'], file_id + '.cmp')
            new_cmp_file = os.path.join(self.temp_dir, file_id + '.cmp')

            DSR.reduce_silence_file(alignment_file_name, ori_cmp_file, new_cmp_file, feat_name='cmp')
            if not self.DRT.check_file_same(ref_cmp_file, new_cmp_file):
                self.logger.info('Difference when generating to file %s' % new_cmp_file)

    def cmp_norm_test(self):
        '''
        cmp_resil --> cmp_resil_norm
        Compare cmp_resil_norm vs current cmp_resil_norm
        Results: Good, 10000 files
            Need to pad end; beginning has sufficient silence
        '''
        cfg = self.cfg
        file_list = self.file_id_list[:10000]

        from frontend_mw545.data_norm import Data_Mean_Var_Normaliser
        DMVN = Data_Mean_Var_Normaliser(cfg)
        norm_info_file = cfg.nn_feat_resil_norm_files['cmp']
        DMVN.load_mean_std_values(norm_info_file)

        for file_id in file_list:
            
            ori_cmp_file = os.path.join(cfg.nn_feat_resil_dirs['cmp'], file_id + '.cmp')
            ref_cmp_file = os.path.join(cfg.nn_feat_resil_norm_dirs['cmp'], file_id + '.cmp')
            new_cmp_file = os.path.join(self.temp_dir, file_id + '.cmp')

            DMVN.norm_file(ori_cmp_file, new_cmp_file, feat_name='cmp')
            if not self.DRT.check_file_same(ref_cmp_file, new_cmp_file):
                self.logger.info('Difference when generating to file %s' % new_cmp_file)

    def compute_normaliser_test(self):
        '''
        Result: Good
            Almost the same as current norm_info_file, precision difference
            All files stored float32 for storage reduction; freshly computed values are float64
            Time: 1min13s, 30s (why std faster?)
        '''
        cfg = self.cfg

        from frontend_mw545.data_norm import Data_Mean_Var_Normaliser
        DMVN = Data_Mean_Var_Normaliser(cfg)

        self.logger.info('start computing mean')
        DMVN.compute_mean()
        self.logger.info('start computing std')
        DMVN.compute_std()

        ref_norm_info_file = cfg.nn_feat_resil_norm_files['cmp']
        new_norm_info_file = os.path.join(self.temp_dir, 'cmp_info.dat')

        DMVN.save_mean_std_values(new_norm_info_file)
        if not self.DRT.check_file_same(ref_norm_info_file, new_norm_info_file):
            self.logger.info('Difference when generating to file %s' % new_norm_info_file)

    def cmp_denorm_test(self):
        '''
        cmp_resil_norm --> cmp_resil
        Compare cmp_resil vs current cmp_resil
        Results: Good
            Difference e-7
        '''
        cfg = self.cfg
        file_list = self.file_id_list[:10000]

        from frontend_mw545.data_norm import Data_Mean_Var_Normaliser
        DMVN = Data_Mean_Var_Normaliser(cfg)
        norm_info_file = cfg.nn_feat_resil_norm_files['cmp']
        DMVN.load_mean_std_values(norm_info_file)

        for file_id in file_list:
            ori_cmp_file = os.path.join(cfg.nn_feat_resil_norm_dirs['cmp'], file_id + '.cmp')
            ref_cmp_file = os.path.join(cfg.nn_feat_resil_dirs['cmp'], file_id + '.cmp')
            new_cmp_file = os.path.join(self.temp_dir, file_id + '.cmp')

            DMVN.denorm_file(ori_cmp_file, new_cmp_file, feat_name='cmp')
            if not self.DRT.check_file_same(ref_cmp_file, new_cmp_file):
                self.logger.info('Difference when generating to file %s' % new_cmp_file)


class Data_Waveform_Pipeline_Test(Data_Test_Base):
    """docstring for Data_Vocoder_Pipeline_Test"""
    def __init__(self, cfg=None):
        self.logger = make_logger("Waveform_Test")
        super(Data_Waveform_Pipeline_Test, self).__init__(cfg)

    def test(self):
        self.wav_io_test()
        self.clear_temp_dir()
        
    def wav_io_test(self):
        '''
        wav --> data1 --> wav1 --> data2
        Compare data1 vs data2
        Result: Good
        '''
        cfg = self.cfg
        file_list = self.file_id_list[:1000]

        for file_id in file_list:
            wav_file_0 = os.path.join(cfg.wav_dir, file_id + '.wav')
            wav_data_1 = self.DF_IO.read_wav_2_wav_1D_data(in_file_name=wav_file_0)
            wav_file_1 = os.path.join(self.temp_dir, file_id + '.wav')
            self.DF_IO.write_wav_1D_data_2_wav(wav_data_1, wav_file_1)
            wav_data_2 = self.DF_IO.read_wav_2_wav_1D_data(in_file_name=wav_file_1)

            # self.logger.info('checking files %s %s'% (wav_file_0, wav_file_1))
            # self.DRT.check_file_same(wav_file_0, wav_file_1) # Header contains Nan, don't run this
            if not self.DRT.check_data_same(wav_data_1, wav_data_2):
                self.logger.info('Different data %s %s'% (wav_file_0, wav_file_1))    
            

####################
# Tools often used #
####################

class Build_Error_Accuracy_Plotter(object):
    """
    Extract errors from log files
    Plot 3 losses / accuracy
    """
    def __init__(self):
        super().__init__()
        self.graph_plotter   = Graph_Plotting()
        self.log_file_reader = Build_Log_File_Reader()

    def plot(self):
        # self.error_plot()
        # self.vuv_plot()
        self.pos_plot_cosine()

        # self.k_final_plot()
        # self.k_variation_plot()
        # self.tau_variation_plot()

        # self.tacotron_loss_plot()
        # log_file_name = '/home/dawna/tts/mw545/TorchDV/dv_cmp_baseline/run_grid.sh.o6092852'
        # fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/dv_cmp_accu_win.png'
        # self.plot_accuracy_single_file_window(log_file_name, fig_file_name)

    def error_plot(self):
        legend_name_dict = {
          'vocoder_1': '/home/dawna/tts/mw545/TorchDV/dv_cmp_baseline/old/dv_y_cmp_lr_0.000010_LRe2048B_LRe2048B_Lin2048B_DV2048S4B161T40D3440/run_log_2020_11_28_03_58_17.log',
          'vocoder_2': '/home/dawna/tts/mw545/TorchDV/dv_cmp_baseline/run_grid.sh.o5896039',
          'wav_DNN_1': '/home/dawna/tts/mw545/TorchDV/dv_wav_subwin/old/dv_y_wav_lr_0.000010_DW380_LRe2048B_LRe2048B_Lin2048B_DV2048S4B161M33T640/run_grid.sh.o5891636',
          'wav_DNN_2': '/home/dawna/tts/mw545/TorchDV/dv_wav_subwin/run_grid.sh.o5896042',
          'wav_SincNet_1': '/home/dawna/tts/mw545/TorchDV/dv_wav_sincnet/old/dv_y_wav_lr_0.000010_Sin80_LRe2048B_LRe2048B_Lin2048B_DV2048S4B161M33T640/run_grid.sh.o5892093',
          'wav_SincNet_2': '/home/dawna/tts/mw545/TorchDV/dv_wav_sincnet/run_grid.sh.o5896040',
          'wav_SineNet_1': '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v1/old/fixed_k/run_grid.sh.o5891633',
          'wav_SineNet_2': '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v1/run_grid.sh.o5896043'
          }
        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/test_accu.png'
        self.plot_accuracy_multiple_files(legend_name_dict, fig_file_name, tvt_list=['test'])
        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/test_loss.png'
        self.plot_error_multiple_files(legend_name_dict, fig_file_name, tvt_list=['test'])
        
    def vuv_plot(self):
        l_list = [] # loss
        a_list = [] # accuracy
        m_list = [] # model name
        l_list.append([1.428874, 1.2925487, 1.2612373, 1.3883814, 1.5256463, 1.1116034, 1.1712322, 0.9525329, 0.8777724, 0.8542981, 0.6858293, 0.60121346, 0.62091124, 0.7954179, 0.600099, 0.6762689, 0.42582694, 0.43280134, 0.7068934, 0.5898805, 0.3371754, 0.46046382, 0.437548, 0.4397845, 0.3947994, 0.41314363, 0.3513951, 0.35080868, 0.27368638, 0.4652518, 0.32021454, 0.33949044, 0.4662969, 0.33031774, 0.46056363, 0.33225125, 0.3675128, 0.28929886, 0.2785749, 0.3204605, 0.35214046])
        a_list.append([0.717398592021455, 0.7773722627737226, 0.7950530035335689, 0.7804878048780488, 0.7942708333333334, 0.8228438228438228, 0.7830188679245284, 0.808695652173913, 0.8210332103321033, 0.8466898954703833, 0.8569725864123957, 0.8635321100917431, 0.8884422110552764, 0.8777089783281734, 0.8963117606123869, 0.8729805013927576, 0.9097435897435897, 0.9038084020416176, 0.8819469669451507, 0.9060248853962017, 0.9224368499257058, 0.9231791600212653, 0.9145064805583251, 0.9114383699818439, 0.9311312607944733, 0.9289767526999817, 0.9240336777650211, 0.9257963542625668, 0.9430651484039574, 0.9189847009735744, 0.9301115241635688, 0.9278135657446003, 0.9066118789689951, 0.9263157894736842, 0.9143960523492812, 0.9315967810249894, 0.9182140621740038, 0.932339911397503, 0.9403389830508475, 0.928386092371562, 0.9273065938159692])
        m_list.append('vocoder')

        l_list.append([3.756336, 3.4280255, 3.4028263, 3.2152455, 3.1569912, 2.696796, 2.6975327, 2.6026604, 2.3070982, 2.2808692, 2.2720382, 2.0960417, 1.9825351, 1.7841625, 1.6803299, 2.1205833, 1.7467872, 1.8014351, 1.452399, 1.5371802, 1.3765916, 1.3516349, 1.3549058, 1.2537196, 1.3178142, 1.344342, 1.2987605, 1.3650228, 1.1989025, 1.196669, 1.3127553, 1.0154117, 0.97483045, 1.2725321, 1.2656623, 1.0605351, 0.9594421, 1.165811, 0.97058004, 1.0928574, 0.91379195, 1.1308514, 0.9043078, 1.0267632, 0.9017768, 1.0747843, 1.1208034, 1.0844188, 0.94035727, 0.98120654, 0.95810556, 1.0891197, 0.9860638, 0.9915483, 0.9995197, 0.9683091, 0.92254215, 0.9096437, 1.0345393, 0.8152093, 0.9891143, 0.96490824, 0.96306145, 0.9516142, 0.79731363, 0.83537704, 0.74644494, 0.87676513, 0.73453814, 0.8373775, 0.77076924, 0.8166881, 0.7730807, 0.6853963, 1.0189562, 0.8979759, 0.75534624, 0.89668083])
        a_list.append([0.19478008699855, 0.24311926605504589, 0.24766355140186916, 0.2692307692307692, 0.3198757763975155, 0.40390879478827363, 0.3435374149659864, 0.41690962099125367, 0.4426229508196721, 0.49076517150395776, 0.4830097087378641, 0.5255905511811023, 0.5362563237774031, 0.5840130505709625, 0.5700787401574803, 0.5939553219448095, 0.6405286343612335, 0.6388028895768834, 0.6670092497430626, 0.6402814423922604, 0.6521460602178091, 0.6848341232227488, 0.6815602836879433, 0.6841085271317829, 0.6839729119638827, 0.6813324952859836, 0.7091737150292778, 0.6996606334841629, 0.7221669980119284, 0.7200591424346969, 0.70042194092827, 0.7573560767590618, 0.7592190889370932, 0.6946502057613169, 0.7016976556184317, 0.7381258023106547, 0.754730713245997, 0.7241071428571428, 0.7592307692307693, 0.7523701175578309, 0.7607599844901124, 0.7471910112359551, 0.7799227799227799, 0.7517469657962487, 0.7868203096867122, 0.7392, 0.7585803432137286, 0.7653415669625627, 0.773382777091059, 0.7763649962602842, 0.7769485903814262, 0.7554368561617704, 0.7846464646464646, 0.7853991175290814, 0.7635288767621646, 0.7647058823529411, 0.7834080717488789, 0.7778709136630344, 0.7859213250517598, 0.7942563223317617, 0.7778262767350502, 0.7841239109390126, 0.7809394760614273, 0.781985670419652, 0.799248523886205, 0.7922272047832586, 0.8046744574290484, 0.7916194790486977, 0.8182825484764543, 0.791005291005291, 0.8099520383693045, 0.8050694025347013, 0.821520618556701, 0.831304347826087, 0.7680826636050516, 0.8156748911465893, 0.8011265164644714, 0.795263135393955])
        m_list.append('wav_DNN')

        l_list.append([2.6925933, 2.2654593, 2.1656303, 2.023262, 1.8260854, 1.7033445, 1.7634736, 1.4188303, 1.1854087, 1.2118907, 1.0938692, 0.99949324, 0.78678, 0.98761773, 0.9400676, 0.8552937, 0.73432744, 1.0828032, 0.6916746, 0.6399689, 0.6134341, 0.5388195, 0.65791, 0.7619903, 0.5124296, 0.7575136, 0.7175916, 0.5581941, 0.5844099, 0.4203266, 0.6318817, 0.46323258, 0.51941645, 0.5963128, 0.65569323, 0.42631075, 0.4786926, 0.54417723, 0.48532358, 0.40444687, 0.49541637, 0.40799066, 0.44355714, 0.49257764, 0.45310184, 0.5009011, 0.45841715, 0.5113098, 0.4553478, 0.44126382, 0.4026247, 0.49410668, 0.37898856, 0.43641728, 0.40486526, 0.38113382, 0.34572464, 0.5291834, 0.4175769, 0.36607698, 0.3775524, 0.38536257, 0.36247712, 0.31083164, 0.3229388, 0.35513726, 0.3525698, 0.35386643, 0.3097477, 0.40316975, 0.39938158, 0.3965007, 0.31155178, 0.427028, 0.37835753, 0.35915294, 0.2994597, 0.4685368])
        a_list.append([0.403093281778637, 0.43577981651376146, 0.5093457943925234, 0.48717948717948717, 0.5993788819875776, 0.6026058631921825, 0.6122448979591837, 0.6588921282798834, 0.7423887587822015, 0.712401055408971, 0.7839805825242718, 0.7716535433070866, 0.8161888701517707, 0.7585644371941273, 0.7905511811023622, 0.7950065703022339, 0.8176211453744493, 0.7574819401444789, 0.8324768756423433, 0.8443271767810027, 0.8571428571428571, 0.8570300157977883, 0.8539007092198582, 0.8333333333333334, 0.873589164785553, 0.8390949088623507, 0.863370201691607, 0.8687782805429864, 0.8588469184890656, 0.8940364711680631, 0.8720112517580872, 0.8823027718550107, 0.86941431670282, 0.8588477366255144, 0.8379143088116411, 0.8853230637569534, 0.8853711790393013, 0.8584821428571429, 0.8819230769230769, 0.8972317026924536, 0.8743699108181465, 0.8943820224719101, 0.8907335907335907, 0.8775285031261493, 0.8833273316528628, 0.8748, 0.8790951638065523, 0.8876881512929371, 0.8722702925422332, 0.8859386686611818, 0.8905472636815921, 0.8801983975581839, 0.8981818181818182, 0.8808664259927798, 0.889949977262392, 0.9003143242029636, 0.9035874439461884, 0.8751047778709137, 0.8927536231884058, 0.9052721817402486, 0.9100829332169358, 0.9051306873184899, 0.907859078590786, 0.9216990788126919, 0.9082125603864735, 0.9083208769307424, 0.9081803005008348, 0.9037372593431483, 0.9191135734072022, 0.9042328042328043, 0.9016786570743405, 0.8986119493059747, 0.9181701030927835, 0.8939130434782608, 0.904707233065442, 0.9090469279148524, 0.919844020797227, 0.8893591552306442])
        m_list.append('wav_SincNet')

        l_list.append([4.065203, 3.8629725, 3.4799645, 3.4644885, 3.4549284, 2.9117973, 2.7676492, 2.771489, 2.4369795, 2.535832, 2.1729484, 2.0657763, 2.513563, 1.7812346, 1.9921476, 1.7406453, 2.2242167, 1.811038, 1.379665, 1.566421, 1.8254187, 1.3465531, 1.3660938, 1.4170456, 1.3805223, 1.1728238, 1.2613639, 1.2968718, 1.2942176, 1.1214727, 1.0997443, 1.1221452, 0.9413928, 0.9191902, 0.91536087, 0.9228407, 0.892052, 0.8803783, 0.93113256, 1.053338, 0.92432475, 0.9209549, 0.81303185, 0.8011307, 0.94917864, 0.76154417, 1.0140609, 0.92635244, 0.7427671, 0.887539, 0.9404976, 0.82376343, 0.8335256, 0.706724, 0.88338304, 0.8115657, 0.8102218, 0.7099471, 0.76182145, 0.66384125, 0.82527846, 0.700139, 0.91715854, 0.7629577, 0.8359536, 0.62062645, 0.69432044, 0.82908946, 0.6408153, 0.58017707, 0.77869135, 0.6158277, 0.6569703, 0.6063724, 0.6492218, 0.7250767, 0.47722194, 0.65351427])
        a_list.append([0.1321894635089415, 0.1651376146788991, 0.18691588785046728, 0.2564102564102564, 0.2453416149068323, 0.31596091205211724, 0.3469387755102041, 0.3760932944606414, 0.4309133489461358, 0.41424802110817943, 0.45145631067961167, 0.5019685039370079, 0.47554806070826305, 0.5758564437194127, 0.5228346456692914, 0.6268068331143233, 0.5674008810572687, 0.6109391124871001, 0.6659815005138746, 0.6297273526824978, 0.6374119154388213, 0.707740916271722, 0.6829787234042554, 0.6802325581395349, 0.6811512415349887, 0.7089880578252671, 0.7234873129473, 0.707579185520362, 0.7291252485089463, 0.7407589945786102, 0.7618377871542429, 0.7505330490405118, 0.7956616052060738, 0.7633744855967078, 0.7663702506063056, 0.785622593068036, 0.787117903930131, 0.790625, 0.7611538461538462, 0.7656427758816837, 0.7844125630089182, 0.8089887640449438, 0.8065637065637066, 0.8120632585509379, 0.7792581922938423, 0.8228, 0.7800312012480499, 0.7896565032805867, 0.8273588792748249, 0.7946896035901272, 0.806799336650083, 0.8172453262113697, 0.8076767676767677, 0.8419574809466506, 0.7989995452478399, 0.8302649303996408, 0.8134529147982063, 0.8235540653813914, 0.819047619047619, 0.8324046292327475, 0.8136185072020952, 0.8368828654404646, 0.7917795844625113, 0.8285568065506653, 0.8099838969404187, 0.8599900348779272, 0.8358375069560379, 0.8142695356738392, 0.8520775623268698, 0.8650793650793651, 0.8333333333333334, 0.8569704284852142, 0.8556701030927835, 0.8573913043478261, 0.8478760045924225, 0.8427672955974843, 0.8851819757365684, 0.8487024924116113])
        m_list.append('wav_SineNet_v1')

        l_list.append([2.5138447, 2.0882032, 2.1578147, 1.9637399, 1.6776366, 1.5754153, 1.4376912, 1.3380951, 1.0298268, 0.99254805, 1.2349956, 0.97632486, 0.8356191, 0.8381318, 1.2541744, 1.0081327, 0.75525314, 0.89988655, 0.95494026, 0.66898257, 0.729506, 0.60185164, 0.72104657, 0.90985405, 0.68476236, 0.6139622, 0.9321956, 0.603104, 0.72488713, 0.5829, 0.6863481, 0.57116365, 0.5469955, 0.6770928, 0.6881724, 0.54641616, 0.5840335, 0.5797601, 0.603639, 0.5994683, 0.6306318, 0.5877367, 0.5624198, 0.57936966, 0.5721263, 0.5406084, 0.62329555, 0.6162467, 0.4953895, 0.5271562, 0.5909933, 0.52625453, 0.53620523, 0.63104606, 0.64949656, 0.5033132, 0.4999567, 0.4919997, 0.4844567, 0.4787315, 0.4864495, 0.450883, 0.5647232, 0.5078401, 0.5583005, 0.48201552, 0.60265374, 0.48416626, 0.46002045, 0.48308995, 0.45397824, 0.48773897, 0.46373254, 0.48631403, 0.5822904, 0.4779691, 0.426422, 0.5775577])
        a_list.append([0.4635089415176414, 0.5504587155963303, 0.5514018691588785, 0.5256410256410257, 0.5652173913043478, 0.6254071661237784, 0.6700680272108843, 0.6822157434402333, 0.7423887587822015, 0.7519788918205804, 0.7135922330097088, 0.7677165354330708, 0.7790893760539629, 0.800978792822186, 0.7511811023622047, 0.7739816031537451, 0.8079295154185022, 0.8080495356037152, 0.7841726618705036, 0.8170624450307827, 0.8283151825752723, 0.8412322274881516, 0.8241134751773049, 0.7919896640826873, 0.835214446952596, 0.8372093023255814, 0.8009108653220559, 0.8631221719457014, 0.831013916500994, 0.85066535239034, 0.8265353961556493, 0.8396588486140725, 0.8577006507592191, 0.8333333333333334, 0.8197251414713015, 0.8510911424903723, 0.8449781659388647, 0.85, 0.8565384615384616, 0.8365566932119833, 0.8433501357115161, 0.8524344569288389, 0.8606177606177606, 0.8436925340198602, 0.8631616852718761, 0.8588, 0.858034321372855, 0.8548822848321112, 0.873094355170993, 0.8567688855646971, 0.857379767827529, 0.8691339183517741, 0.8755555555555555, 0.8483754512635379, 0.8422010004547522, 0.8630444544229906, 0.8713004484304933, 0.8646269907795474, 0.8732919254658386, 0.871410201457351, 0.8712352684417285, 0.8760890609874153, 0.8550135501355014, 0.8618219037871033, 0.8663446054750402, 0.8749377179870453, 0.8530884808013356, 0.883352208380521, 0.874792243767313, 0.8751322751322751, 0.8764988009592326, 0.872057936028968, 0.8724226804123711, 0.8759420289855072, 0.8731343283582089, 0.8809869375907112, 0.8964471403812825, 0.8630669915779573])
        m_list.append('wav_SineNet_v2')

        x_list = []
        for l in l_list:
            M = len(l)-1
            x = numpy.arange(M+1) / float(M)
            x_list.append(x)

        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/vuv_test_loss.png'
        self.graph_plotter.single_plot(fig_file_name, x_list, l_list,m_list,title='Test Loss against Level of Voicing', x_label='Voicing %', y_label='CE')
        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/vuv_test_accu.png'
        self.graph_plotter.single_plot(fig_file_name, x_list, a_list,m_list,title='Test Accuracy against Level of Voicing', x_label='Voicing %', y_label='Accuracy')

    def pos_plot_KL(self):
        l_list = [] # loss
        m_list = [] # model name
        l_list.append("0.00032937 0.00035068 0.00044687 0.00045879 0.00040143 0.00044549 0.00056673 0.00063277 0.00061688 0.0006076  0.00062469 0.00060556  0.00055239 0.00056635 0.000613   0.00064    0.00067064 0.00059076 0.00058781 0.00060574 0.00063363 0.0006927  0.00072534 0.00075277  0.00073638 0.00070268 0.00068165 0.00073145 0.00072788 0.00074347 0.00082045 0.00082622 0.00068777 0.00076933 0.00076773 0.00071204 0.00067857 0.00057961 0.00064583 0.00076243 0.00072424 0.00066579 0.00063658 0.00061282 0.00058963 0.00060706 0.00070227 0.00085162 0.00094528 0.00093378")
        m_list.append('vocoder')

        l_list.append("3.38995171e-05 3.47800437e-05 1.03648190e-05 4.16300020e-05 3.63819545e-05 1.38255745e-05 4.05322188e-05 3.59113235e-05 1.12439013e-05 4.09921419e-05 3.81925395e-05 1.51838926e-05 4.34783825e-05 3.92455419e-05 1.71590726e-05 4.61515389e-05 4.26043001e-05 1.58250681e-05 4.61780775e-05 4.36673636e-05 1.83135144e-05 4.51950334e-05 4.00182803e-05 1.97212889e-05 4.39370019e-05 3.97256141e-05 1.67977113e-05 4.21404123e-05 3.94424770e-05 2.02147383e-05 4.73425459e-05 4.17632147e-05 2.48596335e-05 4.84439208e-05 4.21490695e-05 2.08988958e-05 4.31782332e-05 4.40058633e-05 2.24392103e-05 4.79148166e-05 4.60458836e-05 2.57482950e-05 4.99322963e-05 4.62968637e-05 2.24186611e-05 4.48050775e-05 4.88524020e-05 2.50966502e-05 4.73483619e-05 4.95313210e-05")
        m_list.append('wav_SincNet')

        l_list.append("0.00033405 0.00076308 0.0009455  0.00105279 0.00125955 0.00140863 0.00154968 0.00176863 0.00185628 0.00194962 0.00211269 0.00221417 0.00237251 0.00246198 0.00255026 0.00263038 0.00274237 0.00281911 0.00291863 0.00300167 0.00301873 0.00317068 0.00323924 0.00330631 0.00332186 0.00339032 0.00346434 0.00353701 0.00362522 0.00367067 0.00378    0.00386988 0.00393524 0.00396602 0.00398419 0.00407054 0.00416641 0.00421585 0.00424804 0.00423431 0.00433619 0.00437133 0.00437591 0.0043637  0.00442211 0.00448396 0.00461509 0.00457393 0.00463567 0.00464925")
        m_list.append('wav_SineNet_v0')

        l_list.append("0.00043311 0.00073175 0.00088136 0.00091967 0.00104461 0.00116279 0.00125877 0.00138535 0.00145886 0.00158282 0.00174871 0.00180581 0.00187486 0.00199649 0.00209733 0.00214021 0.00229993 0.00228259 0.0022696  0.00232539 0.00235564 0.00243779 0.0025011  0.00257155 0.0026051  0.00283303 0.0028065  0.00276694 0.00278071 0.00273256 0.00276886 0.00274664 0.00281819 0.00288753 0.00287546 0.00286702 0.00290013 0.00300236 0.00307206 0.00311691 0.00298326 0.00300556 0.00320609 0.00303758 0.00310737 0.00317581 0.0033459  0.00331433 0.00335362 0.00341019")
        m_list.append('wav_SineNet_v2')

        x_list = []
        y_list = []
        for l in l_list:
            l_1 = l.split(' ')
            l_new=[float(x) for x in l_1 if len(x)>1]
            print(l_new)
            x = numpy.arange(len(l_new))+1
            x_list.append(x)
            y_list.append(l_new)

        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/pos_test_loss.png'
        self.graph_plotter.single_plot(fig_file_name, x_list, y_list,m_list,title='Distance against Sample Shift', x_label='Sample Shift', y_label='KL Divergence')

    def pos_plot_cosine(self):
        l_list = [] # loss
        m_list = [] # model name
        l_list.append([0.005038160760272445, 0.007097771609965344, 0.008655920002770831, 0.009631910386890293, 0.01054470438975244, 0.011861003976885633, 0.012713511578595953, 0.013449126591101509, 0.014125597334154373, 0.014990042010347465, 0.01556940802817315, 0.015926362094456768, 0.016512530020217332, 0.01726089170628988, 0.017988835807296353, 0.018286953617942157, 0.01887106654493233, 0.018931773611309193, 0.01986698489471506, 0.02042461126853259, 0.02097163656208203, 0.02149595184567788, 0.02188159540582682, 0.022166288410400387, 0.022608406938822174, 0.022732770785779528, 0.022709346507042972, 0.02338117388570781, 0.023698947963337564, 0.02421990474669619, 0.02470378957186105, 0.024990410801061225, 0.025237119228254105, 0.025846972651342358, 0.025930172451062647, 0.026194684433474075, 0.02632351396972189, 0.026789127627095784, 0.027046256078929524, 0.027351572642521988, 0.027557748864613582, 0.027904626667476867, 0.02813160941114705, 0.028320488991445843, 0.02850281463163438, 0.028735891130648682, 0.028912767991981003, 0.029246931710620035, 0.029462010050146772, 0.028653953089282178])
        m_list.append('vocoder')

        l_list.append([0.00341437362281787, 0.007673577785851113, 0.01068440415983571, 0.013625429189254974, 0.014773644097680378, 0.014146327376177688, 0.013341482840406862, 0.011357867253929087, 0.009734411321450129, 0.012243781295739919, 0.015188466832288553, 0.017098506108083313, 0.01893121193212525, 0.019269892957405853, 0.018120155005434523, 0.01684741846766237, 0.01453048873670196, 0.012575948276029713, 0.014464953139654747, 0.016729227833746384, 0.017994181957260134, 0.019172409243883432, 0.018825121374635746, 0.01695241102497672, 0.015023886561205764, 0.012042021274882322, 0.00973445744790355, 0.012491735729867923, 0.01593438998726956, 0.018363114201093777, 0.02076337364907009, 0.02165860128220515, 0.021068385777797537, 0.02034086719315672, 0.018613437869634618, 0.0172124498244901, 0.019372343679141923, 0.02190002990806287, 0.023524476121135045, 0.025086180315711956, 0.02533514395629323, 0.024275548744180035, 0.02312834708178155, 0.021041946802907607, 0.019302258245963228, 0.02103025574826556, 0.023077710317538008, 0.024213043766928177, 0.02525568625831518, 0.024890441804073883])
        m_list.append('wav_SincNet')

        l_list.append([0.004797102939699798, 0.01073969577700829, 0.013370142094256823, 0.015390508561948386, 0.01863291514306373, 0.02110429473546159, 0.0236619892876271, 0.02652389174902997, 0.028882466658024235, 0.03140379285227267, 0.0337841305123872, 0.035366727387081236, 0.03713846898836034, 0.039147872972376055, 0.04074964860846563, 0.04247530462331547, 0.0443411584184171, 0.04574381775323092, 0.04717711176720048, 0.04872312935797257, 0.04985663657348687, 0.05108854897388955, 0.05249561896751781, 0.053526534333478405, 0.05464298647914418, 0.0558685195792632, 0.056832008266039846, 0.057787348096735254, 0.05882470617265783, 0.05965834662859121, 0.060614568273334855, 0.06159736907559605, 0.06238765927125395, 0.06323231699327952, 0.06418699338874531, 0.06488319444137937, 0.06566413323042258, 0.06647003960462661, 0.06714861233252227, 0.06787482329452738, 0.06866556522087205, 0.06930296636615868, 0.06992807069470938, 0.07056076124909097, 0.07118747773730018, 0.07179313971512863, 0.07246864263328258, 0.07309873903212702, 0.07372507656493084, 0.07431622455061265])
        m_list.append('wav_SineNet_v0')

        l_list.append([0.009968847644946464, 0.015137879285658544, 0.016603846558799262, 0.017986597770596028, 0.02135260245122312, 0.024003340627838172, 0.02594298578138163, 0.029021886662590368, 0.031047850701758715, 0.032953425700983814, 0.034547767824657724, 0.03656452635171971, 0.03832582392901891, 0.04044571634797096, 0.04131339988004536, 0.04217500524206582, 0.04475920018841279, 0.04580392559927424, 0.04715644023847746, 0.0490396249053457, 0.048848936586099456, 0.04994339945809216, 0.05205207529559497, 0.05346110883908661, 0.05386827804541797, 0.05556106142150225, 0.05676314770435762, 0.057171674613831415, 0.05824583296626321, 0.05981447351985242, 0.06048810195807768, 0.06111304182197334, 0.06221112031783271, 0.06294639229776339, 0.06428165998861257, 0.064631669687336, 0.06486214914140674, 0.06635167000367859, 0.06706393054501598, 0.06736326125023756, 0.06768769957337337, 0.06818034766229786, 0.06975488824451809, 0.07042314297499912, 0.07084993019295868, 0.0713198897947984, 0.07230802606820337, 0.07303154684333468, 0.0745160736771612, 0.07477174982001299])
        m_list.append('wav_SineNet_v2')

        x_list = []
        y_list = []
        for l in l_list:
            x = numpy.arange(len(l))+1
            x_list.append(x)
            y_list.append(l)

        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/pos_test_cosine_loss.png'
        self.graph_plotter.single_plot(fig_file_name, x_list, y_list,m_list,title='Distance against Sample Shift', x_label='Sample Shift', y_label='Cosine Distance')

    def pos_plot(self):
        l_list = [] # loss
        m_list = [] # model name
        l_list.append([0.005038160760272445, 0.007097771609965344, 0.008655920002770831, 0.009631910386890293, 0.01054470438975244, 0.011861003976885633, 0.012713511578595953, 0.013449126591101509, 0.014125597334154373, 0.014990042010347465, 0.01556940802817315, 0.015926362094456768, 0.016512530020217332, 0.01726089170628988, 0.017988835807296353, 0.018286953617942157, 0.01887106654493233, 0.018931773611309193, 0.01986698489471506, 0.02042461126853259, 0.02097163656208203, 0.02149595184567788, 0.02188159540582682, 0.022166288410400387, 0.022608406938822174, 0.022732770785779528, 0.022709346507042972, 0.02338117388570781, 0.023698947963337564, 0.02421990474669619, 0.02470378957186105, 0.024990410801061225, 0.025237119228254105, 0.025846972651342358, 0.025930172451062647, 0.026194684433474075, 0.02632351396972189, 0.026789127627095784, 0.027046256078929524, 0.027351572642521988, 0.027557748864613582, 0.027904626667476867, 0.02813160941114705, 0.028320488991445843, 0.02850281463163438, 0.028735891130648682, 0.028912767991981003, 0.029246931710620035, 0.029462010050146772, 0.028653953089282178])
        m_list.append('vocoder')

        l_list.append([0.00341437362281787, 0.007673577785851113, 0.01068440415983571, 0.013625429189254974, 0.014773644097680378, 0.014146327376177688, 0.013341482840406862, 0.011357867253929087, 0.009734411321450129, 0.012243781295739919, 0.015188466832288553, 0.017098506108083313, 0.01893121193212525, 0.019269892957405853, 0.018120155005434523, 0.01684741846766237, 0.01453048873670196, 0.012575948276029713, 0.014464953139654747, 0.016729227833746384, 0.017994181957260134, 0.019172409243883432, 0.018825121374635746, 0.01695241102497672, 0.015023886561205764, 0.012042021274882322, 0.00973445744790355, 0.012491735729867923, 0.01593438998726956, 0.018363114201093777, 0.02076337364907009, 0.02165860128220515, 0.021068385777797537, 0.02034086719315672, 0.018613437869634618, 0.0172124498244901, 0.019372343679141923, 0.02190002990806287, 0.023524476121135045, 0.025086180315711956, 0.02533514395629323, 0.024275548744180035, 0.02312834708178155, 0.021041946802907607, 0.019302258245963228, 0.02103025574826556, 0.023077710317538008, 0.024213043766928177, 0.02525568625831518, 0.024890441804073883])
        m_list.append('wav_SincNet')

        l_list.append([0.004797102944515657, 0.01073969594121355, 0.013370142043301926, 0.015390508598455706, 0.01863291506927234, 0.021104294596423077, 0.023661989235740102, 0.026523891744757836, 0.028882466420105262, 0.03140379287573056, 0.03378413056248766, 0.03536672747058202, 0.037138469071239716, 0.039147872992959964, 0.04074964852737278, 0.04247530452676526, 0.04434115826477566, 0.045743817374020855, 0.047177111539922996, 0.04872312929016217, 0.04985663646427562, 0.051088548869882525, 0.05249561896837224, 0.05352653428586354, 0.05464298639207034, 0.055868519557436484, 0.05683200823380466, 0.05778734799195148, 0.05882470603113371, 0.059658346316220606, 0.06061456817620208, 0.06159736892945026, 0.06238765922053208, 0.06323231670452217, 0.06418699341593162, 0.06488319445978838, 0.06566413299813506, 0.06647003964459047, 0.06714861198904278, 0.06787482328314796, 0.06866556506909481, 0.06930296627096778, 0.06992807046428606, 0.07056076121258364, 0.07118747746264087, 0.07179313940069965, 0.07246864256679265, 0.07309873899837717, 0.07372507643455312, 0.07431622457694452])
        m_list.append('wav_SineNet_v0')

        l_list.append([0.00996884774405995, 0.015137879242781864, 0.016603846577285947, 0.017986597736263613, 0.02135260254241374, 0.024003340832279157, 0.025942985480778818, 0.029021886783219868, 0.031047850612432298, 0.03295342587986968, 0.034547767874447494, 0.036564526310784905, 0.03832582371401411, 0.040445716117120424, 0.041313399701470195, 0.0421750050752196, 0.044759200247290555, 0.045803925565563224, 0.04715644000327712, 0.04903962488313061, 0.04884893659767305, 0.04994339946927739, 0.052052075134652064, 0.053461108941850834, 0.053868277947314254, 0.055561061537082866, 0.056763147661325584, 0.05717167432969573, 0.05824583306289109, 0.05981447352948414, 0.060488102075677855, 0.0611130417214617, 0.06221112088719154, 0.06294639215010293, 0.06428165984048606, 0.0646316697476896, 0.06486214921255717, 0.0663516700108247, 0.06706393065958682, 0.06736326095561572, 0.06768769971474214, 0.06818034748201385, 0.06975488807394346, 0.07042314294462813, 0.07084993006906681, 0.07131988975914551, 0.07230802626192519, 0.07303154675929018, 0.07451607387554353, 0.07477174956547153])
        m_list.append('wav_SineNet_v2')

        x_list = []
        y_list = []
        for l in l_list:
            # l_1 = l.split(' ')
            # l_new=[float(x) for x in l_1 if len(x)>1]
            # print(l_new)
            x = numpy.arange(len(l))+1
            x_list.append(x)
            y_list.append(l)

        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/pos_test_loss.png'
        self.graph_plotter.single_plot(fig_file_name, x_list, y_list,m_list,title='Distance against Sample Shift', x_label='Sample Shift', y_label='Cosine Distance')

    def k_final_plot(self):
        k_spacing = 0.2
        k_val = [0.14543846, 0.3890939, 0.5868369, 0.71958566, 0.886157, 1.0916392, 1.4410459, 1.7031391, 1.8446494, 1.9953889, 2.0323706, 2.2944477, 2.7719922, 2.9563646, 3.0923753, 3.2823963, 3.3731978, 3.6154666, 3.871967, 4.0593514, 4.269055, 4.4588323, 4.572425, 4.809374, 4.9531536, 5.2177896, 5.270889, 5.6898355, 5.8052044, 5.984596, 6.2261505, 6.410798, 6.596303, 6.8110485, 7.01098, 7.187664, 7.3190994, 7.583075, 7.7600613, 8.05407, 8.25828, 8.379881, 8.618487, 8.8271055, 9.032815, 9.178591, 9.407661, 9.626593, 9.781006, 9.931776, 10.185048, 10.354397, 10.644824, 10.858389, 10.973687, 11.212779, 11.498183, 11.618048, 11.773966, 12.032457, 12.199862, 12.334483, 12.610323, 12.8205805, 13.002675, 13.1926565, 13.460133, 13.592515, 13.812666, 13.983125, 14.150169, 14.411791, 14.601097, 14.827031, 14.995887, 15.204199, 15.362075, 15.612572, 15.81706, 16.000694, 16.222887, 16.367588, 16.609724, 16.767296, 16.955708, 17.211475, 17.396381, 17.499258, 17.764679, 18.01483, 18.188103, 18.410717, 18.674805, 18.814768, 19.003288, 19.223158, 19.368404, 19.58496, 19.830566, 20.010426, 20.199112, 20.382456, 20.612215, 20.818375, 21.005049, 21.282207, 21.405027, 21.609222, 21.798512, 22.0114, 22.214165, 22.398836, 22.542332, 22.795897, 23.043554, 23.202423, 23.412409, 23.62514, 23.821018, 24.028383, 24.18153, 24.424934, 24.538269, 24.796488, 25.060404, 25.210938, 25.3935, 25.612116, 25.765648, 26.062227, 26.227201, 26.418066, 26.54378, 26.810032, 26.995518, 27.131338, 27.388577, 27.633652, 27.833523, 28.008806, 28.135864, 28.389208, 28.626823, 28.814577, 28.94799, 29.195585, 29.441916, 29.583271, 29.801474, 29.967787, 30.188988, 30.384521, 30.549034, 30.800295, 31.00569, 31.223738, 31.424913, 31.627129, 31.82212, 32.033314]


        x = (numpy.arange(len(k_val))+1)*k_spacing

        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/k_final_variation_sinenet_v1.png'
        self.graph_plotter.single_plot(fig_file_name, [x], [k_val],['gamma_k'],title='Gamma_k against initial values', x_label='gamma_k init', y_label='gamma_k final')

    def k_variation_plot(self):
        log_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v2/run_grid.sh.o5908908'
        best_epoch_num = 120
        K = 64
        K_spacing = 0.5
        K = 160
        K_spacing = 0.2

        k_holder = [[(k+1)*K_spacing] for k in range(K)]

        file_lines = []
        with open(log_file_name) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 1:
                    continue
                file_lines.append(line)

        for i,single_line in enumerate(file_lines):
            if 'Printing gamma_k values' in single_line:
                k_line = file_lines[i+1]
                k_list = k_line.strip().replace('[','').replace(']','').replace(' ','').split(',')
                for k in range(K):
                    k_holder[k].append(float(k_list[k]))

        x_list = numpy.arange(K)+1
        y_list = []

        for k_list in k_holder:
            y_list.append(k_list[best_epoch_num]-k_list[0])

        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/gamma_k_variation.png'
        graph_plotter = Graph_Plotting()

        graph_plotter.single_bar_plot(fig_file_name, x_list, y_list, w=0.6*K_spacing, title='change in gamma_k against k', x_label='k', y_label='change in gamma_k')

        # Too many curves; split into 8 graphs
        legend_list = [(k+1)*K_spacing for k in range(K)]
        num_graphs = 8
        num_curves = int(K / num_graphs)
        for i in range(num_graphs):
            fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/gamma_k_vs_epoch_%i.png' % (i+1)
            graph_plotter = Graph_Plotting()
            graph_plotter.single_plot(fig_file_name, None, k_holder[i*num_curves:(i+1)*num_curves], legend_list[i*num_curves:(i+1)*num_curves], title='gamma_k against Epoch', x_label='Epoch', y_label='gamma_k')

    def tau_variation_plot(self):
        log_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v2/run_grid.sh.o5908908'
        best_epoch_num = 120
        K = 64
        K_spacing = 0.5
        K = 160
        K_spacing = 0.2

        k_holder = [[(k+1)*K_spacing] for k in range(K)]

        file_lines = []
        with open(log_file_name) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 1:
                    continue
                file_lines.append(line)

        for i,single_line in enumerate(file_lines):
            if 'Printing tau_k values' in single_line:
                k_line = file_lines[i+1]
                k_list = k_line.strip().replace('[','').replace(']','').replace(' ','').split(',')
                for k in range(K):
                    k_holder[k].append(float(k_list[k]))

        x_list = numpy.arange(K)+1
        y_list = []

        for k_list in k_holder:
            y_list.append(k_list[best_epoch_num])

        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/tau_variation.png'
        graph_plotter = Graph_Plotting()

        graph_plotter.single_bar_plot(fig_file_name, x_list, y_list, w=0.6*K_spacing, title='tau_k against k', x_label='k', y_label='tau_k')


    def plot_error_single_file(self, log_file_name, fig_file_name):
        train_error, valid_error, test_error = self.log_file_reader.extract_errors_from_log_file(log_file_name)
        self.graph_plotter.single_plot(fig_file_name, None, [train_error, valid_error, test_error], ['train', 'valid', 'test'])

    def plot_error_multiple_files(self, legend_name_dict, fig_file_name, tvt_list=['train', 'valid', 'test']):
        x_list = []
        error_list  = []
        legend_list = []

        for legend_name in legend_name_dict:
            log_file_name = legend_name_dict[legend_name]
            train_error, valid_error, test_error = self.log_file_reader.extract_errors_from_log_file(log_file_name)
            new_legend_name, x = self.check_retrain_make_x(legend_name, train_error)
            x_list.append(x)
            if 'train' in tvt_list:
                error_list.append(train_error)
                legend_list.append(new_legend_name+'_train')
            if 'valid' in tvt_list:
                error_list.append(valid_error)
                legend_list.append(new_legend_name+'_valid')
            if 'test' in tvt_list:
                error_list.append(test_error)
                legend_list.append(new_legend_name+'_test')

        self.graph_plotter.single_plot(fig_file_name, x_list, error_list, legend_list, title='Loss against Epoch', x_label='Epoch', y_label='Loss')

    def plot_accuracy_single_file(self, log_file_name, fig_file_name):
        train_accuracy, valid_accuracy, test_accuracy = self.log_file_reader.extract_accuracy_from_log_file(log_file_name)
        self.graph_plotter.single_plot(fig_file_name, None, [train_accuracy, valid_accuracy, test_accuracy], ['train', 'valid', 'test'])

    def plot_accuracy_single_file_window(self, log_file_name, fig_file_name):
        train_accuracy, valid_accuracy, test_accuracy, train_accuracy_win, valid_accuracy_win, test_accuracy_win = self.log_file_reader.extract_accuracy_win_from_log_file(log_file_name)
        self.graph_plotter.single_plot(fig_file_name, None, [train_accuracy, valid_accuracy, test_accuracy, train_accuracy_win, valid_accuracy_win, test_accuracy_win], ['train', 'valid', 'test', 'train_win', 'valid_win', 'test_win'])

    def plot_accuracy_multiple_files(self, legend_name_dict, fig_file_name, tvt_list=['train', 'valid', 'test']):
        x_list = []
        accuracy_list = []
        legend_list   = []

        for legend_name in legend_name_dict:
            log_file_name = legend_name_dict[legend_name]
            train_accuracy, valid_accuracy, test_accuracy = self.log_file_reader.extract_accuracy_from_log_file(log_file_name)
            new_legend_name, x = self.check_retrain_make_x(legend_name, train_accuracy)
            x_list.append(x)
            if 'train' in tvt_list:
                accuracy_list.append(train_accuracy)
                legend_list.append(new_legend_name+'_train')
            if 'valid' in tvt_list:
                accuracy_list.append(valid_accuracy)
                legend_list.append(new_legend_name+'_valid')
            if 'test' in tvt_list:
                accuracy_list.append(test_accuracy)
                legend_list.append(new_legend_name+'_test')

        self.graph_plotter.single_plot(fig_file_name, x_list, accuracy_list, legend_list, title='Accuracy against Epoch', x_label='Epoch', y_label='Accuracy')

    def check_retrain_make_x(self, legend_name, train_accuracy):
        '''
        Check if it is re-train
        If so, make x to start at particular epoch number
        If not, use None
        Remove the start epoch number from legend name
        '''
        if '_retrain_' in legend_name:
            string_list = legend_name.split('_')
            start_index = int(string_list[string_list.index('retrain')+1])
            x = range(start_index, start_index+len(train_accuracy))
            # Remove the start index from legend name
            del(string_list[string_list.index('retrain')+1])
            new_legend_name = '_'.join(string_list)
            return new_legend_name, x
        else:
            return legend_name, None


    def tacotron_loss_plot(self):
        log_file_name = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/CUED_Tacotron2/run_grid.sh.o6010311'
        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/tacotron_train_valid_loss.png'
        file_lines = []
        with open(log_file_name) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 1:
                    continue
                file_lines.append(line)

        train_loss_batch = {}
        train_loss_epoch = []
        valid_loss_epoch = []

        for single_line in file_lines:
            if 'train_loss' in single_line:
                # DLL 2021-06-08 02:26:16.326406 - (1, 3815) train_loss : 0.7363263368606567
                epoch_num = int( single_line.split('(')[1].split(',')[0] )
                if epoch_num not in train_loss_batch.keys():
                    train_loss_batch[epoch_num] = []
                l = float( single_line.split(' ')[-1] )
                train_loss_batch[epoch_num].append(l)

            if 'val_loss' in single_line:
                # DLL 2021-06-08 00:48:24.526653 - (0,) val_loss : 0.7507863759994506
                l = float( single_line.split(' ')[-1] )
                valid_loss_epoch.append(l)

        for i in range(len(valid_loss_epoch)):
            train_loss_epoch.append(numpy.mean(train_loss_batch[i]))

        self.graph_plotter.single_plot(fig_file_name, None, [train_loss_epoch, valid_loss_epoch], ['train', 'valid', 'test'], title='Loss against Epoch', x_label='Epoch', y_label='Loss')


#########################
# Main function to call #
#########################

def run_Frontend_Test(cfg):
    # from frontend_mw545.data_io import Data_File_Directory_Utils
    # data_list_writer = Data_File_Directory_Utils(cfg)
    # data_list_writer.copy_to_scratch()

    # plot_k_variation()

    plotter = Build_Error_Accuracy_Plotter()
    plotter.plot()

    # test_class = Data_Vocoder_Position_Test(cfg)
    # test_class = Data_Acoustic_Pipeline_Test(cfg)
    # test_class = Data_Vocoder_Position_Test(cfg)
    # test_class.test()
    # test_class.run()
    # temp_test(cfg)