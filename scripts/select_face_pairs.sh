
declare -a datasets=("lfw-deepfunneled")
declare -a datasubsets=("val")

for dataset in "${datasets[@]}"; do
  for datasubset in "${datasubsets[@]}"; do
    echo $dataset
    echo $datasubset
    python scripts/select_face_pairs.py --net vgg_face_dag --dataset $dataset --datasubset $datasubset --use-hard --n-pairs 1000 --n-val 10 --id-thresh 18
  done
done
