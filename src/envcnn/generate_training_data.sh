folder=$(dirname "$0")
#echo $folder
for f in $(find . -name 'feature_*.env')
do
  #echo $f
  python3 /home/xiaowen/code/envcnn/src/envcnn/generate_validation_feature_list.py $f
done
