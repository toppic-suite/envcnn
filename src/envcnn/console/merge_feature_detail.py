#!/usr/bin/env python3

import sys
import csv

valid_file_name = sys.argv[1]
detail_file_name = sys.argv[2]
output_file_name = sys.argv[3]

valid_file = open(valid_file_name, "r")
detail_file = open(detail_file_name, "r")
output_file = open(output_file_name, "w")

valid_reader = csv.reader(valid_file)
detail_reader = csv.reader(detail_file, delimiter="\t")
output_writer = csv.writer(output_file)

detail_dict = {}

for row in detail_reader:
  key = "features_matrix_" + row[0] + ".csv"
  #print(key)
  detail_dict[key] = row

for valid_row in valid_reader:
  detail_row = detail_dict[valid_row[0]] 
  valid_row = valid_row + detail_row
  output_writer.writerow(valid_row)
