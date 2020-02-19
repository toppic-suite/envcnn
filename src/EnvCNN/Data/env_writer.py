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

def write(envelope_list, output_file_name):
  f = open(output_file_name, "w")
  for env in envelope_list:
    f.write("BEGIN ENVELOPE" + "\n")
    f.write("SPEC_ID=" + str(env.header.spec_id) + "\n")
    f.write("PEAK_ID=" + str(env.header.peak_id) + "\n")
    f.write("CHARGE=" + str(env.header.charge) + "\n")
    f.write("MONO_MZ=" + str(env.header.mono_mz) + "\n")
    f.write("MONO_MASS=" + str(env.header.mono_mass) + "\n")
    f.write("THEO_INTE_SUM=" + str(env.header.inte_sum) + "\n")
    f.write("BASELINE_INTE=" + str(env.header.base_inte) + "\n")
    f.write("TOPFD_SCORE=" + str(env.header.topfd_score) + "\n")
    f.write("PREDICTION_SCORE=" + str(env.header.pred_score) + "\n")
    f.write("ION_TYPE=" + env.header.ion_type + "\n")
    f.write("LABEL=" + str(env.header.label) + "\n")
    f.write("Theoretical Peak MZ values and Intensities" + "\n")
    for peak in env.theo_peak_list:
      f.write(str(peak.mass) + " " + str(peak.intensity) + "\n")
    f.write("Experimental Peak MZ values and Intensities" + "\n")
    for peak in env.exp_peak_list:
      f.write(str(peak.mass) + " " + str(peak.intensity) + "\n")
    #f.write("Experimental Peak MZ values and Intensities in interval [minimum_theo_env_peak-0.1, maximum_theo_env_peak+0.1]" + "\n")
    #for peak in env.intv_peak_list:
    #  f.write(str(peak.mass) + " " + str(peak.intensity) + "\n")
    f.write("END ENVELOPE" + "\n")
    f.write("\n")
  f.close()
