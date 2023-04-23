#!/usr/bin/env python3

import sys
import csv

valid_file_name = sys.argv[1]
predict_file_name = sys.argv[2]
output_file_name = sys.argv[3]

valid_file = open(valid_file_name, "r")
predict_file = open(predict_file_name, "r")
output_file = open(output_file_name, "w")

valid_reader = csv.reader(valid_file)
predict_reader = csv.reader(predict_file)
output_writer = csv.writer(output_file)

count = 0
for valid_row in valid_reader:
  predict_row = next(predict_reader)
  valid_row = valid_row + predict_row
  if predict_row[0] > predict_row[1]:
    pred = 0 
  else:
    pred = 1
  valid_row.append(pred)
  if int(valid_row[3]) == pred:
    valid_row.append(1)
    count = count + 1
  else:
    valid_row.append(0)
  output_writer.writerow(valid_row)
print("Correct prediction:", count)
