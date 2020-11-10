# frontend_tests.py

import os, sys, pickle, time, shutil, logging, copy 
import math, numpy, scipy
numpy.random.seed(545)

from frontend_mw545.modules import make_logger, read_file_list
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

    def test_cfg_used(self):
        '''
        Make new files and compare with old files
        Result: Good; files of same size; although order of files can be different
        '''
        cfg = self.cfg
        in_file_name = cfg.file_id_list_file['all']
        out_file_name = cfg.file_id_list_file['used'] + '.test'
        not_used_file_name = cfg.file_id_list_file['excluded'] + '.test'
        self.DLFIO.write_file_list_cfg_used(in_file_name, out_file_name, not_used_file_name)

        # used_list_old = self.DLFIO.read_file_list(cfg.file_id_list_file['used'])
        # used_list_new = self.DLFIO.read_file_list(out_file_name)
        # for file_id in used_list_new:
        #     if file_id not in used_list_old:
        #         print(file_id)

    def test_dv_used(self):
        '''
        Make new files and compare with old files
        Result: Good; exactly same files
        '''
        cfg = self.cfg

        in_file_name = cfg.file_id_list_file['all']
        out_file_name = cfg.file_id_list_file['dv_test'] + '.test'

        self.DLFIO.write_file_list_dv_test(in_file_name, out_file_name)

    def run_long_enough(self):
        self.DLFIO.write_file_list_long_enough()


class Data_File_Converter_Test(object):
    """docstring for Data_File_Converter_Test"""
    def __init__(self, cfg=None):
        super(Data_File_Converter_Test, self).__init__()
        self.logger = make_logger("DFC_Test")

        self.cfg = cfg
        self.DIO = Data_File_IO(cfg)
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
            y_data, y_num_frame = self.DIO.load_data_file_frame(y, 1)
            r_data, r_num_frame = self.DIO.load_data_file_frame(r, 1)

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

            pitch_reaper_data = self.DIO.read_pitch_reaper(pitch_reaper_file)
            pitch_16kHz_data, l_16k = self.DIO.load_data_file_frame(pitch_16kHz_file, 1)
            wav_16kHz_data, l_16k = self.DIO.load_data_file_frame(wav_cmp_file, 1)

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

            f0_200Hz_data, l_200 = self.DIO.load_data_file_frame(f0_200Hz_file, 1)
            f0_16kHz_data, l_16k = self.DIO.load_data_file_frame(f0_16kHz_file, 1)

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

            f0_reaper_data = self.DIO.read_f0_reaper(f0_reaper_file)
            f0_200Hz_data, l = self.DIO.load_data_file_frame(f0_200Hz_file, 1)

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
            pitch_list = self.DIO.read_pitch_reaper(pitch_reaper_file)
            wav_data = self.DIO.read_wav_2_wav_1D_data(file_id=file_id)

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
            f0_reaper_data = self.DIO.read_f0_reaper(f0_reaper_file)
            lf0_pml_data   = self.DIO.read_pml_by_name_feat(file_id, 'lf0')
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
            wav_1D_data = self.DIO.read_wav_2_wav_1D_data(wav_file_1)
            self.DIO.write_wav_1D_data_2_wav(wav_1D_data, wav_file_2, sample_rate=self.cfg.wav_sr)

    def wav_cut_short_test(self):
        '''
        1. Convert wav file to data
        2. Remove a few samples
        3. Convert back to wav file
        4. Convert all wav files to vocoder features
        5. Compare
        '''
        pass

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

                cmp_data_1, frame_number_1 = self.DIO.load_data_file_frame(cmp_file_1, cfg.nn_feature_dims['cmp'])
                cmp_data_2, frame_number_2 = self.DIO.load_data_file_frame(cmp_file_2, cfg.nn_feature_dims['cmp'])

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
                            data_1 = self.DIO.read_pml_by_name_feat(file_id, feat_name)
                            file_name_2 = os.path.join(test_temp_dir, file_id + cfg.acoustic_file_ext_dict[feat_name])
                            data_2, frame_number = self.DIO.load_data_file_frame(file_name_2, cfg.acoustic_in_dimension_dict[feat_name])
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

                cmp_data_1, frame_number_1 = self.DIO.load_data_file_frame(cmp_file_1, cfg.nn_feature_dims['cmp'])
                cmp_data_2, frame_number_2 = self.DIO.load_data_file_frame(cmp_file_2, cfg.nn_feature_dims['cmp'])

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
            data_1 = self.DIO.read_pml_by_name_feat(file_id, feat_name)
            file_name_2 = os.path.join(test_temp_dir, file_id + cfg.acoustic_file_ext_dict[feat_name])
            data_2, frame_number = self.DIO.load_data_file_frame(file_name_2, cfg.acoustic_in_dimension_dict[feat_name])
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

            lf0_data, num_samples = self.DIO.load_data_file_frame(lf0_file, 1)
            lf016k_data, num_samples = self.DIO.load_data_file_frame(lf016k_file, 1)
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
        self.DIO = Data_File_IO(cfg)
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
        self.DIO = Data_File_IO(cfg)

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

class Data_Loader_Test(object):
    """docstring for Data_Loader_Test"""
    def __init__(self, cfg=None):
        super(Data_Loader_Test, self).__init__()
        self.cfg = cfg
        self.logger = make_logger("Data_Loader_Test")

        self.cfg = cfg
        self.DIO = Data_File_IO(cfg)

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

        file_id_list_file = self.cfg.file_id_list_file['used']
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

                self.DRT.check_data_dict_same(ref_feed_dict, new_feed_dict)

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

class Build_Error_Plotter(object):
    """
    Extract errors from log files
    Plot 3 losses / accuracy
    """
    def __init__(self):
        super().__init__()
        self.graph_plotter   = Graph_Plotting()
        self.log_file_reader = Build_Log_File_Reader()

    def plot_error_single_file(self, log_file_name, fig_file_name):
        train_error, valid_error, test_error = self.log_file_reader.extract_errors_from_log_file(log_file_name)
        self.graph_plotter.single_plot(fig_file_name, None, [train_error, valid_error, test_error], ['train', 'valid', 'test'])

    def plot_error_multiple_files(self, legend_name_dict, fig_file_name, tvt_list=['train', 'valid', 'test']):
        error_list  = []
        legend_list = []

        for legend_name in legend_name_dict:
            log_file_name = legend_name_dict[legend_name]
            train_error, valid_error, test_error = self.log_file_reader.extract_errors_from_log_file(log_file_name)
            if 'train' in tvt_list:
                error_list.append(train_error)
                legend_list.append(legend_name+'_train')
            if 'valid' in tvt_list:
                error_list.append(valid_error)
                legend_list.append(legend_name+'_valid')
            if 'test' in tvt_list:
                error_list.append(test_error)
                legend_list.append(legend_name+'_test')

        self.graph_plotter.single_plot(fig_file_name, None, error_list, legend_list)

    def plot_accuracy_single_file(self, log_file_name, fig_file_name):
        train_accuracy, valid_accuracy, test_accuracy = self.log_file_reader.extract_accuracy_from_log_file(log_file_name)
        self.graph_plotter.single_plot(fig_file_name, None, [train_accuracy, valid_accuracy, test_accuracy], ['train', 'valid', 'test'])

    def plot_accuracy_multiple_files(self, log_file_name_list, legend_name_list, fig_file_name):
        accuracy_list = []
        legend_list   = []

        for log_file_name, legend_name in zip(log_file_name_list, legend_name_list):
            train_accuracy, valid_accuracy, test_accuracy = self.log_file_reader.extract_accuracy_from_log_file(log_file_name)
            accuracy_list.extend([train_accuracy, valid_accuracy, test_accuracy])
            legend_list.extend([legend_name+'_train', legend_name+'_valid', legend_name+'_test'])

        self.graph_plotter.single_plot(fig_file_name, None, accuracy_list, legend_list)

def temp_test():
    x = numpy.random.rand(640)
    w_s = numpy.random.rand(16,640)
    w_c = numpy.random.rand(16,640)
    w_s_a = numpy.random.rand(86,16)
    w_c_a = numpy.random.rand(86,16)

    w_s_c = numpy.concatenate((w_s, w_c), axis=0)
    w_a = numpy.concatenate((w_s_a, w_c_a), axis=1)

    print(w_s_c.shape)
    print(w_a.shape)

    d = numpy.dot

    h_1 = d(w_s_a, d(w_s, x)) + d(w_c_a, d(w_c, x))
    h_2 = d(w_a, d(w_s_c, x))

    print(h_1.shape)
    print(h_2.shape)


    print((h_1==h_2).all())
    print(h_1-h_2)



#######################
# Common tools to use #
#######################

class Data_Replicate_Test(object):
    """ 
    Make a few files in a new test directory
    Compare with files already generated
    """
    def __init__(self, cfg=None):
        super(Data_Replicate_Test, self).__init__()
        self.cfg = cfg
        self.logger = make_logger("Replicate_Test")

        self.DIO = Data_File_IO(cfg)

    def check_data_same(self, data_1, data_2, l_1=None, l_2=None):
        if l_1 is None:
            l_1 = data_1.shape[0]
        if l_2 is None:
            l_2 = data_2.shape[0]

        if l_1 != l_2:
            self.logger.info('Different Files Lengths! %i %i' % (l_1, l_2))
            return False
        if (data_1 == data_2).all():
            return True
        else:
            self.logger.info('Different Data!')
            data_3 = data_1[data_1 != data_2] - data_2[data_1 != data_2]
            # print(data_3)
            self.logger.info('Max Difference is %s' % str(numpy.max(data_3)))
            return False

    def check_file_same(self, file_1, file_2):
        '''
        Return True if the 2 files are exactly the same:
        1. same length
        2. same values
        '''
        data_1, l_1 = self.DIO.load_data_file_frame(file_1, 1)
        data_2, l_2 = self.DIO.load_data_file_frame(file_2, 1)

        return self.check_data_same(data_1, data_2, l_1, l_2)

    def check_file_dict_same(self, file_dict_1, file_dict_2):
        bool_all_same = True
        for k in file_dict_1:
            if self.check_file_same(file_dict_1[k], file_dict_2[k]):
                continue
            else:
                self.logger.info('Data of key %s is different' % k)
                bool_all_same = False
        return bool_all_same

    def check_data_dict_same(self, data_dict_1, data_dict_2):
        bool_all_same = True
        for k in data_dict_1:
            if self.check_data_same(data_dict_1[k], data_dict_2[k]):
                continue
            else:
                self.logger.info('Data of key %s is different' % k)
                bool_all_same = False
        return bool_all_same

class Graph_Plotting(object):
    """
    Functions for plotting
    """
    def __init__(self):
        super().__init__()
        self.logger = make_logger("Graph_Plot")

        import matplotlib.pyplot as plt
        self.plt = plt

    def change_default_x_list(self, x_list, y_list):
        '''
        Make x-axis if there is none
        '''
        l = len(y_list)

        if x_list is None:
            x_list = [None] * l

        for i in range(l):
            if x_list[i] is None:
                x_list[i] = range(len(y_list[i]))

        return x_list
        
    def single_plot(self, fig_file_name, x_list, y_list, legend_list, title=None, x_label=None, y_label=None):
        '''
        Line plots
        Plot multiple lines on the same graph
        '''
        x_list = self.change_default_x_list(x_list, y_list)
        fig, ax = self.plt.subplots()
        for x, y, l in zip(x_list, y_list, legend_list):
            if l is None:
                ax.plot(x, y)
            else:
                ax.plot(x, y, label=l)

        self.set_title_labels(ax, title, x_label, y_label)

        self.logger.info('Saving to %s' % fig_file_name)
        fig.savefig(fig_file_name, format="png")
        self.plt.close(fig)

    def one_line_one_scatter(self, fig_file_name, x_list, y_list, legend_list, title=None, x_label=None, y_label=None):
        '''
        One line plot, one scatter plot
        Useful for data samples on a curve
        '''
        x_list = self.change_default_x_list(x_list, y_list)
        fig, ax = self.plt.subplots()
        ax.plot(x_list[0], y_list[0], label=legend_list[0])
        ax.scatter(x_list[1], y_list[1], label=legend_list[1], c='r', marker='.')

        self.set_title_labels(ax, title, x_label, y_label)
        
        self.logger.info('Saving to %s' % fig_file_name)
        fig.savefig(fig_file_name, format="png")
        self.plt.close(fig)

    def set_title_labels(self, ax, title, x_label, y_label):
        '''
        Set title and x/y labels for the graph
        To be used for all plotting functions
        '''
        if title is not None:
            ax.set_title(title)  # Add a title to the axes
        if x_label is not None:
            ax.set_xlabel(x_label)  # Add a title to the axes
        if y_label is not None:
            ax.set_ylabel(y_label)  # Add a title to the axes
        ax.legend()

class Build_Log_File_Reader(object):
    """
    Functions for reading log files e.g. errors
    """
    def __init__(self):
        super().__init__()
        pass

    def extract_errors_from_log_file(self, log_file_name):
        file_lines = []
        with open(log_file_name) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 1:
                    continue
                file_lines.append(line)

        train_error = []
        valid_error = []
        test_error  = []

        for single_line in file_lines:
            words = single_line.strip().split(' ')
            if 'epoch' in words and 'loss:' in words:
                # words_new = words
                if 'train' in words:
                    train_index = words.index('train')+1
                elif 'training' in words:
                    train_index = words.index('training')+1
                if 'validation' in words:
                    valid_index = words.index('validation')+1
                elif 'valid' in words:
                    valid_index = words.index('valid')+1
                test_index  = words.index('test')+1
                train_error.append(float(words[train_index][:-1]))
                valid_error.append(float(words[valid_index][:-1]))
                test_error.append(float(words[test_index][:-1]))
                
        return (train_error, valid_error, test_error)


    def extract_accuracy_from_log_file(self, log_file_name):
        file_lines = []
        with open(log_file_name) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 1:
                    continue
                file_lines.append(line)

        train_accuracy = []
        valid_accuracy = []
        test_accuracy  = []

        for single_line in file_lines:
            words = single_line.strip().split(' ')
            if 'epoch' in words and 'accu:' in words:
                # words_new = words
                if 'train' in words:
                    train_index = words.index('train')+1
                elif 'training' in words:
                    train_index = words.index('training')+1
                if 'validation' in words:
                    valid_index = words.index('validation')+1
                elif 'valid' in words:
                    valid_index = words.index('valid')+1
                test_index  = words.index('test')+1
                train_accuracy.append(float(words[train_index][:-1]))
                valid_accuracy.append(float(words[valid_index][:-1]))
                test_accuracy.append(float(words[test_index][:-1]))
                
        return (train_accuracy, valid_accuracy, test_accuracy)
        

        

#########################
# Main function to call #
#########################

def run_Frontend_Test(cfg):
    # from frontend_mw545.data_io import Data_File_Directory_Utils
    # data_list_writer = Data_File_Directory_Utils(cfg)
    # data_list_writer.copy_to_scratch()

    # DLFIO_Test = Data_List_File_IO_Test(cfg)
    # DLFIO_Test.run_long_enough()

    # DFC_Test = Data_File_Converter_Test(cfg)
    # DFC_Test.temp_test()

    # DSR_test = Data_Silence_Reducer_Test(cfg)
    # DSR_test.silence_reducer_test()

    # DWMMN_test = Data_Wav_Min_Max_Normaliser_test(cfg)
    # DWMMN_test.normaliser_test()

    # NPU_Test = Numpy_Pytorch_Unfold_Test()
    # NPU_Test.speed_test()

    # DL_Test = Data_Loader_Test(cfg)
    # DL_Test.single_file_data_loader_test()
    # DL_Test.multi_files_data_loader_test_1()
    # DL_Test.multi_files_data_loader_test_2()
    # DL_Test.dv_y_train_data_loader_test_2()

    # shortest_files(cfg)
    # longest_files(cfg)
    
    # Error_Plotter = Build_Error_Plotter()
    # log_file_name_list = ['/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v1/run_grid.sh.o5865954',
    #     '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v2/run_log_2020_08_23_13_38_12.log',
    #     '/home/dawna/tts/mw545/TorchDV/dv_wav_subwin/run_grid.sh.o5865927',
    #     '/home/dawna/tts/mw545/TorchDV/dv_cmp_baseline/run_grid.sh.o5865958']
    # legend_name_list = ['sine_v1','sine_v2','DNN','Vocoder']
    # fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/loss_vs_epoch.png'
    # Error_Plotter.plot_error_multiple_files(log_file_name_list, legend_name_list, fig_file_name)
    # fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/accu_vs_epoch.png'
    # Error_Plotter.plot_accuracy_multiple_files(log_file_name_list, legend_name_list, fig_file_name)

    # temp_test()

    Error_Plotter = Build_Error_Plotter()
    legend_name_dict = {
      # 'Sine_8_Concat': '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v1/old/concat/dv_y_wav_lr_0.000100_Sin86f16_LRe256B_LRe256B_Lin8B_DV8S8B154M40T640/run_grid.sh.o5866734',
      # 'Sine_4_Concat_f32': '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v1/old/concat/dv_y_wav_lr_0.000100_Sin86f32_LRe256B_LRe256B_Lin8B_DV8S4B154M40T640/run_grid.sh.o5867368',
      # 'Sine_8_Add': '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v1/run_grid.sh.o5867966',
      'Sine_4_Add': '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v1/run_grid.sh.o5867967',
      # 'DNN_8_Concat': '/home/dawna/tts/mw545/TorchDV/dv_wav_subwin/old/concat/dv_y_wav_lr_0.000100_LRe86_LRe256B_LRe256B_Lin8B_DV8S8B154M40T640/run_grid.sh.o5867856',
      # 'DNN_4_Concat': '/home/dawna/tts/mw545/TorchDV/dv_wav_subwin/old/concat/dv_y_wav_lr_0.000100_LRe86_LRe256B_LRe256B_Lin8B_DV8S4B154M40T640/run_grid.sh.o5867864',
      # 'DNN_8_Add': '/home/dawna/tts/mw545/TorchDV/dv_wav_subwin/run_grid.sh.o5867975',
      'DNN_4_Add': '/home/dawna/tts/mw545/TorchDV/dv_wav_subwin/run_grid.sh.o5867998',
      'SinRes_4_Add': '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v2/run_grid.sh.o5867988',
    }
    fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/4_add_loss_vs_epoch.png'
    Error_Plotter.plot_error_multiple_files(legend_name_dict, fig_file_name)
    # fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/sinenet_test/4_add_valid_loss_vs_epoch.png'
    # Error_Plotter.plot_error_multiple_files(legend_name_dict, fig_file_name, tvt_list=['valid'])