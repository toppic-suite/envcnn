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
import math
import EnvCNN.Data.env_util as EnvUtil
from EnvCNN.Data.feature import Feature

class Envelope:
  def __init__(self, header, theo_peak_list, exp_peak_list, min_exp_mz, feature_list):
    self.header = header
    self.theo_peak_list = theo_peak_list
    self.exp_peak_list = exp_peak_list
    self.min_exp_mz = min_exp_mz
    self.feature_list = feature_list
    # self.intv_peak_list = intv_peak_list

  @classmethod
  def get_env(cls, header, theo_peak_list, exp_peak_list):
    min_exp_mz = ""
    feature = ""
    return cls(header, theo_peak_list, exp_peak_list, min_exp_mz, feature)

  def get_header(self):
    return self.header

  def get_intv_peak_list(self):
    intv_peak_list = []
    min_theo_peak = min(self.theo_peak_list, key=lambda x: x.mass).mass
    max_theo_peak = max(self.theo_peak_list, key=lambda x: x.mass).mass
    for peak in self.exp_peak_list:
      if ((peak.mass >= min_theo_peak - 0.1) and (peak.mass <= max_theo_peak + 0.1)):
        intv_peak_list.append(peak)
    self.exp_peak_list = intv_peak_list

  def get_peak_feature_list(self, tolerance):
    feature_list = []
    self.min_exp_mz = self.header.mono_mz - 0.1
    ## get the max_theo_inetensity and Normalize the intensities
    max_theo_intensity = EnvUtil.get_max_intensity(self.theo_peak_list)
    log_theo_intensity = math.log10(max_theo_intensity)
    log_base_intenisty = math.log10(self.header.base_inte)
    charge = self.header.charge
    for peak in self.theo_peak_list:
      bin_index = EnvUtil.get_bin_indx(peak.mass, self.min_exp_mz)
      norm_theo_peak_inte = EnvUtil.get_normalized_intensity(peak.intensity, max_theo_intensity)
      exp_peak_idx = EnvUtil.find_exp_peak_idx(peak, self.exp_peak_list, tolerance)
      if exp_peak_idx >= 0:
        exp_peak = self.exp_peak_list[exp_peak_idx]
        norm_exp_peak_inte = EnvUtil.get_normalized_intensity(exp_peak.intensity, max_theo_intensity)
        mass_diff = peak.mass - exp_peak.mass
        inte_diff = norm_theo_peak_inte - norm_exp_peak_inte
      else:
        norm_exp_peak_inte = 0
        mass_diff = -peak.mass
        inte_diff = norm_theo_peak_inte - norm_exp_peak_inte
      feature = Feature(bin_index, norm_theo_peak_inte, norm_exp_peak_inte, charge, 
                            mass_diff, inte_diff, log_theo_intensity, log_base_intenisty)
      feature_list.append(feature)
    self.feature_list = feature_list

  def getMatrix(self):
    matrix = numpy.zeros(shape=(300, 6))
    max_theo_intensity = EnvUtil.get_max_intensity(self.theo_peak_list)
    for feature in self.feature_list:
      EnvUtil.populate_matrix(feature, matrix, self.header.mono_mass) 
      
    for peak in self.exp_peak_list:
      pos = EnvUtil.get_bin_indx(peak.mass, self.min_exp_mz)
      ## Evaluate Peak Condition
      peak_condition = False
      for i in range(0, 3):
	  ## to accomodate +2 and -2 bins - reason tolerance of 0.02. We have already selected the best peak - with min mass_diff
        if pos + i < 300 and pos - i > -1 and matrix[pos][0] == 0: 
          if matrix[pos - i][0] == 0 and matrix[pos + i][0] == 0:
            peak_condition = True
          else:
            peak_condition = False
            break
      ## populate noise peaks
      if peak_condition == True:
        matrix[pos][1] = EnvUtil.get_normalized_intensity(peak.intensity, max_theo_intensity)
    return matrix
