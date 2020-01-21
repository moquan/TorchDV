# First half is feature vector; second half is std vector

from io_funcs.binary_io import  BinaryIOCollection
import os, numpy
import logging

class   SplitFilesWithStd(object):

    def __init__(self, feature_dimension):
        self.feature_dimension = feature_dimension
        
        
    def split_files_with_std(self, in_file_list, out_file_list, out_std_file_list):

        logger = logging.getLogger('split_files_with_std')

        logger.debug('split_files_with_std for %d files' % len(in_file_list) )

        io_funcs = BinaryIOCollection()

        findex=0
        flen=len(in_file_list)

        for file_name in in_file_list:
            
            feature_with_log_var, num_frames = io_funcs.load_binary_file_frame(file_name, self.feature_dimension*2)
            logger.info('processing %4d of %4d: %s' % (findex,flen,file_name) )
            # features_only = numpy.zeros(num_frames*self.feature_dimension)
            # std_only = numpy.zeros(num_frames*self.feature_dimension)

            # for idx in xrange(num_frames):
            #     features_only[idx*self.feature_dimension:(idx+1)*self.feature_dimension] = feature_with_log_var[idx*2*self.feature_dimension:(idx*2+1)*self.feature_dimension]
            #     std_only[idx*self.feature_dimension:(idx+1)*self.feature_dimension] = feature_with_log_var[(idx*2+1)*self.feature_dimension:(idx*2+2)*self.feature_dimension]

            features_only = feature_with_log_var[:,:self.feature_dimension]
            # TODO: exponential here if output is log variance
            std_only = numpy.exp(feature_with_log_var[:,self.feature_dimension:]/2.)
            # std_only = feature_with_log_var[:,self.feature_dimension:]

            io_funcs.array_to_binary_file(features_only, out_file_list[findex])
            io_funcs.array_to_binary_file(std_only, out_std_file_list[findex])

            findex=findex+1