#!/bin/bash
folder=$(dirname "$0")
echo $folder
for f in $(find . -name '*sp_*.xml'); 
  do 
    filename=$(basename $f .xml)
    echo $filename
    python3 ${folder}/../Exec/add_anno.py ${filename}.env ${filename}.xml 
  done

