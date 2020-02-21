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

#!/usr/bin/env python3

import os
import sys
from shutil import copyfile

filename = sys.argv[1]
file = open(filename)
all_lines = file.readlines()
file.close()

## Assign file name to header
filename = os.path.basename(filename)
if len(all_lines) < 2:
  exit()
  
for i in range(len(all_lines)):
  line = all_lines[i]
  mono = line.split('=')
  ## Reads Scan Number
  if("SPEC_ID" in line):
    spec_id = int(mono[1][0:len(mono[1])-1])
    break

file_name = "sp_" + str(spec_id) + ".env"
#file_name = os.path.basename(os.getcwd()) + "_sp_" + str(spec_id) + ".env"
copyfile(sys.argv[1], file_name)
