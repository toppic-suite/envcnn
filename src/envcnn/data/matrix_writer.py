##Copyright (c) 2014 - 2020, The Trustees of Indiana University.
##
##Licensed under the Apache License, Version 2.0 (the "License");
##you may not use this file except in compliance with the License.
##You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##Unless required by applicable law or agreed to in writing, software
##distributed under the License is distributed on an "AS IS" BASIS,
##WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##See the License for the specific language governing permissions and
##limitations under the License.

#!/usr/bin/python3

import numpy
import os

def write(env, matrix, file_prefix):
  matrix_directory = os.path.join(os.path.dirname(os.getcwd()), "TrainData")
  if os.path.isdir(matrix_directory ) == False:
    os.mkdir(matrix_directory )

  _write_label_names(env, matrix_directory, file_prefix)
  _write_parameters(env, matrix_directory, file_prefix)
  output_file_name = file_prefix + "_matrix_" + str(env.header.spec_id) + "_" + str(env.header.peak_id) + ".csv"
  output_file = os.path.join(matrix_directory, output_file_name)
  numpy.savetxt(output_file, matrix, fmt='%.16f', delimiter=',', newline='\n')

def _write_label_names(env, matrix_directory, file_prefix):
  filename = matrix_directory + os.sep + "label.csv"
  f = open(filename , "a+")
  f.write(file_prefix + "_matrix_" + str(env.header.spec_id) + "_" + str(env.header.peak_id) + ".csv" + "," + str(env.header.label) + "\n")
  f.close()

def _write_parameters(env, matrix_directory, file_prefix):
  filename = matrix_directory + os.sep + "parameters.csv"
  f = open(filename , "a+")
  matrix_name = file_prefix + "_matrix_" + str(env.header.spec_id) + "_" + str(env.header.peak_id) + ".csv"
  topfd_score = env.header.topfd_score
  ion_type = env.header.ion_type
  label = env.header.label
  f.write( matrix_name + "," + str(topfd_score) + "," + str(ion_type) + "," + str(label) + "\n")
  f.close()

def _write_label(env, matrix_directory):
  filename = matrix_directory + os.sep + "labels.csv"
  f = open(filename , "a+")
  f.write(str(env.header.label) + "\n")
  f.close()
