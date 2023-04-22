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

import os
from EnvCNN.Data.peak import Peak
from EnvCNN.Data.env_header import EnvHeader
from EnvCNN.Data.envelope import Envelope

def _read_header(env_lines):
  peak_id = -1
  spec_id = -1
  topfd_score = -1
  charge = -1
  mono_mass = -1
  mono_mz = -1
  base_inte = -1
  inte_sum = -1

  for i in range(len(env_lines)):
    line = env_lines[i]
    mono = line.split('=')
    ## Reads Peak ID of all envelopes corresponding to a scan
    if("PEAK_ID" in line):
      peak_id = int(mono[1][0:len(mono[1])-1])
    ## Reads Spectrum ID
    if("SPEC_ID" in line):
      spec_id = int(mono[1][0:len(mono[1])-1])
    ## Reads TopFD Score of all envelopes corresponding to a scan
    if("SCORE" == line[0:5]):
      topfd_score = float(mono[1][0:len(mono[1])-1])
    ## Read Charge of all envelopes corresponding to a scan
    if("CHARGE" in line):
      charge = int(mono[1][0:len(mono[1])-1])
    ## Read theoretical monoisotopic mass all envelopes corresponding to a scan
    if("THEO_MONO_MASS" in line):
      mono_mass = float(mono[1][0:len(mono[1])-1])
    ## Read theoretical monoisotopic mass all envelopes corresponding to a scan
    if("THEO_MONO_MZ" in line):
      mono_mz = float(mono[1][0:len(mono[1])-1])
    if("THEO_INTE_SUM" in line):
      inte_sum = float(mono[1][0:len(mono[1])-1])
    ## Reads Baseline Intensity of the scan
    if("BASELINE_INTE" in line):
      base_inte = float(mono[1][0:len(mono[1])-1])
  header = EnvHeader.get_header(spec_id, base_inte, peak_id, 
          charge, mono_mz, mono_mass, inte_sum, topfd_score)
  return header

def _read_theo_peaks(env_lines):
  theo_line = "Theoretical Peak MZ values and Intensities"
  exp_line = "Experimental Peak MZ values and Intensities"
  peak_list = []
  i = 0
  while(env_lines[i].strip() != theo_line): 
    i = i + 1
  i = i + 1
  while(env_lines[i].strip() != exp_line): 
    mono = env_lines[i].split(' ')
    mass = float(mono[0])
    intensity = float(mono[1][0:len(mono[1])-1])
    peak = Peak(mass,intensity)
    peak_list.append(peak)
    i = i + 1
  return peak_list

def _read_exp_peaks(env_lines):
  exp_line = "Experimental Peak MZ values and Intensities"
  end_line = "END ENVELOPE"
  peak_list = []
  i = 0
  while(env_lines[i].strip() != exp_line): 
    i = i + 1
  i = i + 1
  while(env_lines[i].strip() != end_line): 
    mono = env_lines[i].split(' ')
    mass = float(mono[0])
    intensity = float(mono[1][0:len(mono[1])-1])
    peak = Peak(mass,intensity)
    peak_list.append(peak)
    i = i + 1
  return peak_list

def _parse_envelope(env_lines):
  header = _read_header(env_lines)
  theo_peak_list = _read_theo_peaks(env_lines)
  exp_peak_list = _read_exp_peaks(env_lines)
  env = Envelope.get_env(header, theo_peak_list, exp_peak_list)
  return env

def _get_end_index(all_lines, begin_idx):
  idx = begin_idx
  while (idx < len(all_lines) and "END ENVELOPE" not in all_lines[idx]):
    idx = idx + 1
  return idx

def read_env_file(filename):
  file = open(filename)
  all_lines = file.readlines()
  file.close()

  ## Assign file name to header
  filename = os.path.basename(filename)
  if len(all_lines) < 2:
    return

  env_list = []
  begin_idx = 0
  while (begin_idx < len(all_lines)):
    end_idx = _get_end_index(all_lines, begin_idx)
    env_lines = all_lines[begin_idx:end_idx +1]
    begin_idx = end_idx + 1
    if begin_idx >= len(all_lines):
      break
    env = _parse_envelope(env_lines)
    env.header.file_name = filename
    env_list.append(env)
  return env_list
