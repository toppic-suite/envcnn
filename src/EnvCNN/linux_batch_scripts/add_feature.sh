folder=$(dirname "$0")
echo $folder
for f in $(find . -name 'annotated_*.env') 
do 
  echo $f
  python3 ${folder}/../Exec/add_feature.py $f 
done
