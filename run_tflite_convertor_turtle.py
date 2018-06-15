#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: run_tflte_convertor_turtle.py

    description:

        -  To convert tensorflow frozen graph to  tflite format

    references:
        - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/mobile/tflite/devguide.md#2-convert-the-model-format
        - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/g3doc/cmdline_examples.md#savedmodel

    author: Jaewook Kang and Doyoung Kwak
    date  : 2018 June
'''

import sys
from os import getcwd
sys.path.insert(0,  getcwd()+'/convertor/')
from tflite_convertor import TFliteConvertor

# your frozen graph pb
input_frozen_pb_path    = getcwd()+'/pb_and_ckpt/turtle/frozen_pb_out/'
sys.path.insert(0,  input_frozen_pb_path)

# your dir for exporting tflite file
output_tflite_path   = getcwd()+'/pb_and_ckpt/turtle/tflite_out/'


# your dir path for tensorflow source
# where you need to fork tensorflow repo
#
PATH_TENSORFLOW_SRC = '/Users/jwkangmacpro2/SourceCodes/tensorflow/'


# The output/input node names are obtained from Tensorboard
output_node_names   = 'HG/last5/Sigmoid'
input_node_names    =  'input_image'

# input placeholder shape
# -----------------------------------
# Comment from jwkanggist 180615
# see here Doyoung
# At this point you need to specify the shape of the input images
# shape = [batchsize, height, weight, channel num]
# In this project, we will use 256X256X3 images as input
# where 256x256 = height x weight and the extra x3 is from RGB scale of the images.
# Note that in the lenet5 example, we have used the MNIST dataset which has gray scale
# such that we use x1 at the same point.
input_shape_str = '1,256,256,3'
#---------------------------------------



tflite_convertor = TFliteConvertor()

# tflite config
tflite_convertor.set_config_for_tflite(input_dir_path   =input_frozen_pb_path,
                                    output_dir_path     =output_tflite_path,
                                    input_pb_file       ='frozen_net.pb',
                                    output_tflite_file  ='tflite_net.tflite',
                                    inference_type      ='FLOAT',
                                    input_shape         = input_shape_str,
                                    input_array         = input_node_names,
                                    output_array        = output_node_names,
                                    tf_src_dir_path     = PATH_TENSORFLOW_SRC)

# frozen grpah to tflite conversion
tflite_convertor.convert_to_tflite_from_frozen_graph()
