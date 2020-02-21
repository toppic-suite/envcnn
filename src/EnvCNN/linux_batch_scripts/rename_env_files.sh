folder=$(dirname "$0")
echo $folder
for f in $(find . -name '*_ms2.env');
  do
    python3 ${folder}/../Exec/change_env_name.py $f;
  done