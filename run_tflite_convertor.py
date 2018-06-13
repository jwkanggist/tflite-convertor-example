#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: run_tflte_convertor.py

    description:

        -  To convert tensorflow frozen graph to  tflite format

    references:
        - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/mobile/tflite/devguide.md#2-convert-the-model-format
        - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/g3doc/cmdline_examples.md#savedmodel

    author: Jaewook Kang
    date  : 2018 Apr
'''

import sys
from os import getcwd
sys.path.insert(0,  getcwd()+'/convertor/')
from tflite_convertor import TFliteConvertor

# your frozen graph pb
input_frozen_pb_path    = getcwd()+'/pb_and_ckpt/lenet5/frozen_pb_out/'
sys.path.insert(0,  input_frozen_pb_path)

# your dir for exporting tflite file
output_tflite_path   = getcwd()+'/pb_and_ckpt/lenet5/tflite_out/'


# your dir path for tensorflow source
# where you need to fork tensorflow repo
#
PATH_TENSORFLOW_SRC = '/Users/jwkangmacpro2/SourceCodes/tensorflow/'


# The output/input node names are obtained from Tensorboard
output_node_names   = 'model_out/Softmax'
input_node_names    =  'input'

# input placeholder shape
input_shape_str = '1,28,28,1'



tflite_convertor = TFliteConvertor()

# tflite config
tflite_convertor.set_config_for_tflite(input_dir_path   =input_frozen_pb_path,
                                    output_dir_path     =output_tflite_path,
                                    input_pb_file       ='frozen_tf_graph_def_lenet5.pb',
                                    output_tflite_file  ='tflite_lenet5.tflite',
                                    inference_type      ='FLOAT',
                                    input_shape         = input_shape_str,
                                    input_array         = input_node_names,
                                    output_array        = output_node_names,
                                    tf_src_dir_path     = PATH_TENSORFLOW_SRC)

# frozen grpah to tflite conversion
tflite_convertor.convert_to_tflite_from_frozen_graph()
