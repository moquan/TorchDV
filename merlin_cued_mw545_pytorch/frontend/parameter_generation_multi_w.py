################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://svn.ecdf.ed.ac.uk/repo/inf/dnn_tts/
#                
#                Centre for Speech Technology Research                 
#                     University of Edinburgh, UK                       
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.                           
#                                                                       
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute  
#  this software and its documentation without restriction, including   
#  without limitation the rights to use, copy, modify, merge, publish,  
#  distribute, sublicense, and/or sell copies of this work, and to      
#  permit persons to whom this work is furnished to do so, subject to   
#  the following conditions:
#  
#   - Redistributions of source code must retain the above copyright  
#     notice, this list of conditions and the following disclaimer.   
#   - Redistributions in binary form must reproduce the above         
#     copyright notice, this list of conditions and the following     
#     disclaimer in the documentation and/or other materials provided 
#     with the distribution.                                          
#   - The authors' names may not be used to endorse or promote products derived 
#     from this software without specific prior written permission.   
#                                  
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK        
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE     
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       
#  THIS SOFTWARE.
################################################################################

## Added FAST_MLPG as a variable here, in case someone wants to use the slow version, but perhaps we 
## should always use the bandmat version?
FAST_MLPG = True
#io_funcs.

from io_funcs.binary_io import  BinaryIOCollection
import os, numpy, sys
import logging

if FAST_MLPG:
    from mlpg_fast_new import MLParameterGenerationFast as MLParameterGeneration
   # pass
else:
    from mlpg import MLParameterGeneration

class   ParameterGenerationMultiW(object):

    def __init__(self, gen_wav_features = ['mgc', 'lf0', 'bap']):
        self.gen_wav_features = gen_wav_features
        self.inf_float = -1.0e+10

        # not really necessary to have the logger rembered in the class - can easily obtain it by name instead
        # self.logger = logging.getLogger('param_generation')

        self.var = {}
    
    def duration_decomposition(self, in_file_list, dimension, out_dimension_dict, file_extension_dict):

        logger = logging.getLogger('param_generation')

        logger.debug('duration_decomposition for %d files' % len(in_file_list) )

        state_number = 5  ## hard coding, try removing in future?

        if len(out_dimension_dict.keys())>1:
            logger.critical("we don't support any additional features along with duration as of now.")
            sys.exit(1)
        else:
            feature_name = out_dimension_dict.keys()[0]
        
        io_funcs = BinaryIOCollection()
        
        findex=0
        flen=len(in_file_list)
        for file_name in in_file_list:
            
            findex=findex+1
            
            dir_name = os.path.dirname(file_name)
            file_id = os.path.splitext(os.path.basename(file_name))[0]

            features, frame_number = io_funcs.load_binary_file_frame(file_name, dimension)
            gen_features = numpy.int32(numpy.round(features))
            gen_features[gen_features<1]=1

            if dimension > state_number:
                gen_features = gen_features[:, state_number]

            logger.info('processing %4d of %4d: %s' % (findex,flen,file_name) )

            new_file_name = os.path.join(dir_name, file_id + file_extension_dict[feature_name])
            io_funcs.array_to_binary_file(gen_features, new_file_name) 

            logger.debug('wrote to file %s' % new_file_name)

    def acoustic_decomposition(self, in_file_list, dimension, out_dimension_dict, file_extension_dict, var_file_dict, multi_w_values, do_MLPG=True):

        logger = logging.getLogger('param_generation')

        logger.debug('acoustic_decomposition for %d files' % len(in_file_list) )

        self.load_covariance(var_file_dict, out_dimension_dict)

        stream_start_index = {}
        dimension_index = 0
        recorded_vuv = False
        vuv_dimension = None

        for feature_name in out_dimension_dict.keys():
           # if feature_name != 'vuv':
            stream_start_index[feature_name] = dimension_index
           # else:
               # vuv_dimension = dimension_index
               # recorded_vuv = True
            
            dimension_index += out_dimension_dict[feature_name]

        io_funcs = BinaryIOCollection()

        mlpg_algo = MLParameterGeneration()

        findex=0
        flen=len(in_file_list)
        for file_name in in_file_list:
            
            findex=findex+1
            
            dir_name = os.path.dirname(file_name)
            file_id = os.path.splitext(os.path.basename(file_name))[0]

            features, frame_number = io_funcs.load_binary_file_frame(file_name, dimension)
            
            logger.info('processing %4d of %4d: %s' % (findex,flen,file_name) )

            for feature_name in self.gen_wav_features:
                
                logger.debug(' feature: %s' % feature_name)
                
                current_features = features[:, stream_start_index[feature_name]:stream_start_index[feature_name]+out_dimension_dict[feature_name]]
                if FAST_MLPG:
                    ### fast version wants variance per frame, not single global one:
                    var = self.var[feature_name]
                    var = numpy.transpose(numpy.tile(var,frame_number))
                else:
                    var = self.var[feature_name]
                    
                # print  var.shape[1]
                if do_MLPG == False:
                    gen_features = current_features
                else:
                    if feature_name in ['mgc', 'mcep']:
                        static_win_1 = multi_w_values.static_win_mgc_1.astype(float)
                        delta_win_1  = multi_w_values.delta_win_mgc_1.astype(float) 
                        acc_win_1    = multi_w_values.acc_win_mgc_1.astype(float) 
                        current_features_1 = numpy.concatenate((current_features[:,0:20],current_features[:,60:80],current_features[:,120:140]),1)
                        var_1 = numpy.concatenate((var[:,0:20],var[:,60:80],var[:,120:140]),1)
                        static_win_2 = multi_w_values.static_win_mgc_2.astype(float) 
                        delta_win_2  = multi_w_values.delta_win_mgc_2.astype(float) 
                        acc_win_2    = multi_w_values.acc_win_mgc_2.astype(float) 
                        current_features_2 = numpy.concatenate((current_features[:,20:40],current_features[:,80:100],current_features[:,140:160]),1)
                        var_2 = numpy.concatenate((var[:,20:40],var[:,80:100],var[:,140:160]),1)
                        static_win_3 = multi_w_values.static_win_mgc_3.astype(float) 
                        delta_win_3  = multi_w_values.delta_win_mgc_3.astype(float) 
                        acc_win_3    = multi_w_values.acc_win_mgc_3.astype(float) 
                        current_features_3 = numpy.concatenate((current_features[:,40:60],current_features[:,100:120],current_features[:,160:180]),1)
                        var_3 = numpy.concatenate((var[:,40:60],var[:,100:120],var[:,160:180]),1)

                        gen_features_1, gen_features_std_1 = mlpg_algo.generation(current_features_1, var_1, 20, static_win_1, delta_win_1, acc_win_1)
                        gen_features_2, gen_features_std_2 = mlpg_algo.generation(current_features_2, var_2, 20, static_win_2, delta_win_2, acc_win_2)
                        gen_features_3, gen_features_std_3 = mlpg_algo.generation(current_features_3, var_3, 20, static_win_3, delta_win_3, acc_win_3)
                        gen_features     = numpy.concatenate((gen_features_1,gen_features_2,gen_features_3),1)
                        gen_features_std = numpy.concatenate((gen_features_std_1,gen_features_std_2,gen_features_std_3),1)
                    elif feature_name in ['lf0', 'F0']:
                        static_win = multi_w_values.static_win_lf0.astype(float) 
                        delta_win  = multi_w_values.delta_win_lf0.astype(float) 
                        acc_win    = multi_w_values.acc_win_lf0.astype(float) 
                        gen_features, gen_features_std = mlpg_algo.generation(current_features, var, out_dimension_dict[feature_name]/3, static_win, delta_win, acc_win)
                    elif feature_name in ['bap', 'bndap']:
                        static_win = multi_w_values.static_win_bap.astype(float) 
                        delta_win  = multi_w_values.delta_win_bap.astype(float) 
                        acc_win    = multi_w_values.acc_win_bap.astype(float) 
                        gen_features, gen_features_std = mlpg_algo.generation(current_features, var, out_dimension_dict[feature_name]/3, static_win, delta_win, acc_win)

                    
               # else:
                   # self.logger.critical("the dimensions do not match for MLPG: %d vs %d" %(var.shape[1], out_dimension_dict[feature_name]))
                   # raise

                logger.debug(' feature dimensions: %d by %d' %(gen_features.shape[0], gen_features.shape[1]))

                if feature_name in ['lf0', 'F0']:
                    if stream_start_index.has_key('vuv'):
                        vuv_feature = features[:, stream_start_index['vuv']:stream_start_index['vuv']+1]

                        for i in xrange(frame_number):
                            if vuv_feature[i, 0] < 0.5:
                                gen_features[i, 0] = self.inf_float

                new_file_name = os.path.join(dir_name, file_id + file_extension_dict[feature_name])
                new_std_file_name = new_file_name + '_std'

                '''
                if cfg.enforce_silence:
                    label_align_dir = cfg.in_label_align_dir
                    in_f = open(label_align_dir+'/'+file_id+'.lab','r')
                    for j in in_f.readlines():
                        fstr = j.strip().split()
                        ftag = fstr[2]
                        ph = ftag[ftag.index('-')+1:ftag.index('+')]
            
                        ph_start = int(int(fstr[0])*(10**-4)/5)
                        ph_end    = int(int(fstr[1])*(10**-4)/5)
            
                        if ph=='SIL':
                            gen_features[ph_start:ph_end, :] = 0.0
                '''

                io_funcs.array_to_binary_file(gen_features, new_file_name)
                logger.debug(' wrote to file %s' % new_file_name)
                io_funcs.array_to_binary_file(gen_features_std, new_std_file_name)
                logger.debug(' wrote to file %s' % new_std_file_name)


    def load_covariance(self, var_file_dict, out_dimension_dict):

        io_funcs = BinaryIOCollection()
        for feature_name in var_file_dict.keys():
            var_values, dimension = io_funcs.load_binary_file_frame(var_file_dict[feature_name], 1)

            var_values = numpy.reshape(var_values, (out_dimension_dict[feature_name], 1))

            self.var[feature_name] = var_values


class ParameterGenerationMultiWEachDim(ParameterGenerationMultiW):
    pass




if __name__ == '__main__':
    
    in_file_list = ['/afs/inf.ed.ac.uk/group/project/dnn_tts/mtl_dnn/gen/dnn_2500_601_229/hvd_678.cmp']

    out_dimension_dict = { 'mgc' : 150,
                           'lf0' : 3,
                           'vuv' : 1,
                           'bap' : 75}
        
    file_extension_dict = {'mgc' : '.mgc',
                           'lf0' : '.lf0',
                           'vuv' : '.vuv',
                           'bap' : '.bap'}

    var_file_dict  = { 'mgc' : '/afs/inf.ed.ac.uk/group/project/dnn_tts/mtl_dnn/data/var/mgc',
                       'lf0' : '/afs/inf.ed.ac.uk/group/project/dnn_tts/mtl_dnn/data/var/lf0',
                       'bap' : '/afs/inf.ed.ac.uk/group/project/dnn_tts/mtl_dnn/data/var/bap'}
         
    generator = ParameterGeneration()

    generator.acoustic_decomposition(in_file_list, 229, out_dimension_dict, file_extension_dict, var_file_dict)
