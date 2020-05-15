
# user-defined variables
slurm=0 # whether to use SLURM to parallelize jobs

# setup directories
python scripts/setup_fbf_dirs.py

# initialize an array of commands to execute with SLURM or locally (will take VERY long locally, as it will be run in serial)
# NOTE: to use SLURM, edit run_slurm.sbatch as needed
declare -a cmds=()

###-----------EXP 1: networks trained on faces or objects at a standard dataset size---------------------------------------------------------------
cmd="python scripts/pretrain.py --net vgg16 --num-workers 12 --dataset imagenet-subset --remove-birds-and-cars --batch-size 128 --epochs 100"
cmds+=( "$cmd" )
cmd="python scripts/pretrain.py --net vgg16 --num-workers 12 --dataset vggface2 --dataset-match imagenet-subset --remove-birds-and-cars --remove-face-overlap --batch-size 128 --epochs 100"
cmds+=( "$cmd" )
## optional: full vggface2 pretraining (used for comparions with humans)
#cmd="python scripts/from_scratch.py --net vgg16 --num-workers 12 --dataset vggface2 --remove-face-overlap --batch-size 128 --epochs 100"
#cmds+=( "$cmd" )

###------------EXP 2: networks trained on faces using different fractions of the total identities---------------------------------------------------
declare -a train_fracs=(0.01 0.1 1.0)
for frac in ${train_fracs}; do
  cmd="python scripts/facebyface.py --net vgg16 --start-frac ${frac} --no-grow-data --num-workers 12 --batch-size 128"
  cmds+=( "$cmd" )
done

###---------------------------------------------------RUN ALL COMMANDS----------------------------------------
ii=0
for cmd in "${cmds[@]}"; do
  let ii+=1
  if [ $slurm -eq 1 ]; then
    sbatch --export="COMMAND=$cmd" --job-name sim-$ii --time 12-00:00:00 --output=log/%j.log scripts/run_slurm.sbatch
  else
    $cmd
  fi
done
