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

class Feature:
  def __init__(self, bin_index, norm_theo_peak_inte, norm_exp_peak_inte, 
               charge, mass_diff, inte_diff, log_theo_intensity, log_base_intenisty):
    self.bin_index = bin_index
    self.norm_theo_peak_inte = norm_theo_peak_inte
    self.norm_exp_peak_inte = norm_exp_peak_inte
    self.charge = charge
    self.mass_diff = mass_diff
    self.inte_diff = inte_diff
    self.log_theo_intensity = log_theo_intensity
    self.log_base_intenisty = log_base_intenisty
