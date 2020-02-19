folder=$(dirname "$0")
echo $folder

for f in $(find . -name 'prsm*.xml') 
  do 
    echo $f
    python3 ${folder}/../Exec/change_xml_name.py $f 
  done
