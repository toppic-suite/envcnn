folder=$(dirname "$0")
echo $folder
for f in $(find . -name 'feature_*.env') 
do 
  echo $f
  python3 ${folder}/../Exec/generate_peak_only_training_data.py $f
done
