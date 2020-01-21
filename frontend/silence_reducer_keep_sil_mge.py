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


import  sys, numpy, re, math
from io_funcs.binary_io import BinaryIOCollection

class SilenceReducerMGE(object):
    def __init__(self, n_cmp, silence_pattern=['*-#+*']):
        self.silence_pattern = silence_pattern
        self.silence_pattern_size = len(silence_pattern)
        self.n_cmp = n_cmp
        
    def reduce_silence(self, in_data_list, in_align_list, out_data_list, 
        possible_frame_numbers=[160,200,240,280,320,360,400,440,480,520,560,600,640,680,720,760,800,
        840,880,920,940,1000,1040,1070,1120,1160,1180,1250,1300,1590,1970],sil_pad=5):
        # This function removes silence from before and after the utterance.
        # It doesn't remove siolence in the middle; also it keeps some frames.
        # It tries to keep 20 frames of silence if there are 20.
        file_number = len(in_data_list)
        align_file_number = len(in_align_list)

        if  file_number != align_file_number:
            print   "The number of input and output files does not equal!\n"
            sys.exit(1)
        if  file_number != len(out_data_list):
            print   "The number of input and output files does not equal!\n"
            sys.exit(1)

        io_funcs = BinaryIOCollection()
        for i in xrange(file_number):

            nonsilence_indices = self.load_alignment(in_align_list[i])
            ori_cmp_data = io_funcs.load_binary_file(in_data_list[i], self.n_cmp)
             
            frame_number = ori_cmp_data.size/self.n_cmp
            
            if len(nonsilence_indices) == frame_number:
                print 'WARNING: no silence found!'
                # previsouly: continue -- in fact we should keep non-silent data!

            no_sil_start = nonsilence_indices[0]
            no_sil_end   = nonsilence_indices[-1]
            no_sil_len   = no_sil_end - no_sil_start + 1
            possible_frame_numbers = numpy.array(possible_frame_numbers)
            final_len    = min(possible_frame_numbers[possible_frame_numbers>=(no_sil_len+sil_pad*2)])
            frames_silence_to_keep = int((final_len-(no_sil_len+sil_pad*2))/2)

            sil_pad_last_idx = min(no_sil_end+sil_pad+frames_silence_to_keep, frame_number-1)
            sil_pad_first_idx = sil_pad_last_idx-final_len+1
            # sil_pad_first_idx = no_sil_start - min(frames_silence_to_keep, no_sil_start)

            # nonsilence_indices = range(sil_pad_first_idx, sil_pad_first_idx+final_len)
            nonsilence_indices = range(sil_pad_first_idx, sil_pad_last_idx+1)

            ## if labels have a few extra frames than audio, this can break the indexing, remove them:
            # nonsilence_indices = [ix for ix in nonsilence_indices if ix < frame_number]

            new_cmp_data = ori_cmp_data[nonsilence_indices,]

            io_funcs.array_to_binary_file(new_cmp_data, out_data_list[i])

            win_i = numpy.zeros(final_len)
            sil_pad_after = min(frames_silence_to_keep,frame_number-1-(no_sil_len+sil_pad))
            sil_pad_before = final_len - no_sil_len - 2*sil_pad - sil_pad_after
            win_i[sil_pad_before:sil_pad_before+no_sil_len+sil_pad*2] = 1
            io_funcs.array_to_binary_file(win_i, out_data_list[i]+'.win')


    def check_silence_pattern(self, label):
        label_size = len(label)
        binary_flag = 0
        for si in xrange(self.silence_pattern_size):
            current_pattern = self.silence_pattern[si]
            current_size = len(current_pattern)
            if current_pattern[0] == '*' and current_pattern[current_size - 1] == '*':
                temp_pattern = current_pattern[1:current_size - 1]
                for il in xrange(1, label_size - current_size + 2):
                    if temp_pattern == label[il:il + current_size - 2]:
                        binary_flag = 1
            elif current_pattern[current_size-1] != '*':
                temp_pattern = current_pattern[1:current_size]
                if temp_pattern == label[label_size - current_size + 1:label_size]:
                    binary_flag = 1
            elif current_pattern[0] != '*':
                temp_pattern = current_pattern[0:current_size - 1]
                if temp_pattern == label[0:current_size - 1]:
                    binary_flag = 1
            if binary_flag == 1:
                break
        
        return  binary_flag # one means yes, zero means no

    def load_alignment(self, alignment_file_name):

        base_frame_index = 0
        nonsilence_frame_index_list = []
        fid = open(alignment_file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            start_time = int(temp_list[0])
            end_time = int(temp_list[1])
            full_label = temp_list[2]
            frame_number = int((end_time - start_time)/50000)

            label_binary_flag = self.check_silence_pattern(full_label)

            if label_binary_flag == 0:
                for frame_index in xrange(frame_number):
                    nonsilence_frame_index_list.append(base_frame_index + frame_index)
            base_frame_index = base_frame_index + frame_number
#            print   start_time, end_time, frame_number, base_frame_index
        fid.close()
        
        return  nonsilence_frame_index_list

#    def load_binary_file(self, file_name, dimension):
        
#        fid_lab = open(file_name, 'rb')
#        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
#        fid_lab.close()
#        features = features[:(dimension * (features.size / dimension))]
#        features = features.reshape((-1, dimension))
        
#        return  features


def trim_silence(in_list, out_list, in_dimension, label_list, label_dimension, \
                                       silence_feature_index, percent_to_keep=0):
    '''
    Function to trim silence from binary label/speech files based on binary labels.
        in_list: list of binary label/speech files to trim
        out_list: trimmed files
        in_dimension: dimension of data to trim
        label_list: list of binary labels which contain trimming criterion
        label_dimesion:
        silence_feature_index: index of feature in labels which is silence: 1 means silence (trim), 0 means leave.
    '''
    assert len(in_list) == len(out_list) == len(label_list)
    io_funcs = BinaryIOCollection()
    for (infile, outfile, label_file) in zip(in_list, out_list, label_list):
    
        data = io_funcs.load_binary_file(infile, in_dimension)
        label = io_funcs.load_binary_file(label_file, label_dimension)
        
        audio_label_difference = data.shape[0] - label.shape[0]
        assert math.fabs(audio_label_difference) < 3,'%s and %s contain different numbers of frames: %s %s'%(infile, label_file,  data.shape[0], label.shape[0])
        
        ## In case they are different, resize -- keep label fixed as we assume this has
        ## already been processed. (This problem only arose with STRAIGHT features.)
        if audio_label_difference < 0:  ## label is longer -- pad audio to match by repeating last frame:
            print 'audio too short -- pad'
            padding = numpy.vstack([data[-1, :]] * int(math.fabs(audio_label_difference)))
            data = numpy.vstack([data, padding])
        elif audio_label_difference > 0: ## audio is longer -- cut it
            print 'audio too long -- trim'
            new_length = label.shape[0]
            data = data[:new_length, :]
        #else: -- expected case -- lengths match, so do nothing
                    
        silence_flag = label[:, silence_feature_index]
#         print silence_flag
        if not (numpy.unique(silence_flag) == numpy.array([0,1])).all():
            ## if it's all 0s or 1s, that's ok:
            assert (numpy.unique(silence_flag) == numpy.array([0]).all()) or \
                   (numpy.unique(silence_flag) == numpy.array([1]).all()), \
                   'dimension %s of %s contains values other than 0 and 1'%(silence_feature_index, infile)
        print 'Remove %d%% of frames (%s frames) as silence... '%(100 * numpy.sum(silence_flag / float(len(silence_flag))), int(numpy.sum(silence_flag)))
        non_silence_indices = numpy.nonzero(silence_flag == 0)  ## get the indices where silence_flag == 0 is True (i.e. != 0)
        if percent_to_keep != 0:
            assert type(percent_to_keep) == int and percent_to_keep > 0
            #print silence_flag
            silence_indices = numpy.nonzero(silence_flag == 1)            
            ## nonzero returns a tuple of arrays, one for each dimension of input array
            silence_indices = silence_indices[0]
            every_nth = 100  / percent_to_keep
            silence_indices_to_keep = silence_indices[::every_nth]  ## every_nth used +as step value in slice
                        ## -1 due to weird error with STRAIGHT features at line 144:
                        ## IndexError: index 445 is out of bounds for axis 0 with size 445 
            if len(silence_indices_to_keep) == 0:
                silence_indices_to_keep = numpy.array([1]) ## avoid errors in case there is no silence
            print '   Restore %s%% (every %sth frame: %s frames) of silent frames'%(percent_to_keep, every_nth, len(silence_indices_to_keep))

            ## Append to end of utt -- same function used for labels and audio
            ## means that violation of temporal order doesn't matter -- will be consistent.
            ## Later, frame shuffling will disperse silent frames evenly across minibatches:
            non_silence_indices = ( numpy.hstack( [non_silence_indices[0], silence_indices_to_keep] ) ) 
                                                    ##  ^---- from tuple and back (see nonzero note above)
        
        trimmed_data = data[non_silence_indices, :]  ## advanced integer indexing
        io_funcs.array_to_binary_file(trimmed_data, outfile)  
    

if  __name__ == '__main__':
    
    cmp_file_list_name = ''
    lab_file_list_name = ''
    align_file_list_name = ''

    n_cmp = 229
    n_lab = 898

    in_cmp_list = ['/group/project/dnn_tts/data/nick/nn_cmp/nick/herald_001.cmp']
    in_lab_list = ['/group/project/dnn_tts/data/nick/nn_new_lab/herald_001.lab']
    in_align_list = ['/group/project/dnn_tts/data/cassia/nick_lab/herald_001.lab']
    
    out_cmp_list = ['/group/project/dnn_tts/data/nick/nn_new_lab/herald_001.tmp.cmp']
    out_lab_list = ['/group/project/dnn_tts/data/nick/nn_new_lab/herald_001.tmp.no.lab']
    
    remover = SilenceRemover(in_cmp_list, in_align_list, n_cmp, out_cmp_list)
    remover.remove_silence()

