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

import sys
import os
import xml.etree.ElementTree as ET
from shutil import copyfile

tree = ET.parse(sys.argv[1])
spec_id = tree.find('./ms/ms_header/ids')

file_name = os.path.basename(os.getcwd()) + "_sp_" + spec_id.text + ".xml"
copyfile(sys.argv[1], file_name)
