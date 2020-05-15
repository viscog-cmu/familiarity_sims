
# user-defined variables
slurm=1 # whether to use SLURM to parallelize jobs

#---------------Setup datasets for familiarization experiments----------------
python scripts/setup/setup_dataset.py --data-loc $data_loc --dataset lfw --id-thresh 18 --n-val 10
python scripts/setup/setup_dataset_bootstrap.py --data-loc $data_loc

# initialize an array of commands to execute with SLURM or locally (will take VERY long locally, as it will be run in serial)
declare -a cmds=()

###-----------------EXP 1: effects of domain of prior experience on novel familiarization-----------------------------------
declare -a nets=('vgg16_train-vggface2-match-imagenet-subset' 'vgg16_train-imagenet-subset' 'vgg16_random' 'vgg16_train-vggface2')
for net in ${nets[@]}; do
  cmd="python scripts/finetune.py --overwrite --dataset lfw-deepfunneled --net ${net} --epochs 50 --id-thresh 18 --n-val 10"
  cmds+=( "$cmd" )
done

###-----------------EXP 2: effects of extent of prior face experience on novel familiarization-----------------------------
declare -a train_fracs=(0.01 0.1 0.5 0.9 1.0)
for frac in ${train_fracs[@]}; do
  cmd="python scripts/finetune.py --overwrite --dataset lfw-deepfunneled --id-thresh 18 --n-val 10 --net vgg16 --fbf-start-frac ${start_frac} --fbf-no-grow-data --epochs 50"
  cmds+=( "$cmd" )
done

###-----------------EXP 3: effects of extent of familiarization experience on familiarization------------------------------
declare -a nets=('vgg16_train-vggface2-match-imagenet-subset' 'vgg16_train-imagenet-subset' 'vgg16_random' 'vgg16_train-vggface2')
declare -a layers=('conv1' 'conv2' 'conv3' 'conv4' 'conv5' 'fc6' 'fc7' 'fc8')
declare -a ntrains=('1' '10' '50' '100' '200' '400')
declare -a datasets=('vggface2-test_subset-100-400-10-10')

for dataset in ${datasets[@]}; do
  for ntrain in ${ntrains[@]}; do
    for net in ${nets[@]}; do
      for layer in ${layers[@]}; do
        cmd="python scripts/finetune_bootstrap.py --overwrite --dataset ${dataset} --net ${net} --epochs 1000 \
        --first-finetuned-layer ${layer} --n-train ${ntrain} --lr-scheduler plateau --verification-phase test --roc-epochs 0 1 50 --save-dists \
        --splitvals 20"
        cmds+=( "$cmd" )
      done
    done
  done
done

###---------------------------------------------------RUN ALL COMMANDS----------------------------------------
ii=0
for cmd in "${cmds[@]}"; do
  let ii+=1
  if [ slurm -eq 1 ]; then
    sbatch --export="COMMAND=$cmd" --job-name sim-$ii --time 12-00:00:00 --output=log/%j.log scripts/run_slurm.sbatch
  else
    $cmd
  fi
done
