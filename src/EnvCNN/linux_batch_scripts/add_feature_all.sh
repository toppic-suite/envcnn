folder=$(dirname "$0")
echo $folder
for f in $(find . -name 'WH*.env')
do
  echo $f
  python3 ${folder}/../Exec/add_features_all.py $f
done

