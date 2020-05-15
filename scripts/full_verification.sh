
slurm=0

declare -a cmds=()
declare -a subs=('11' '12' '13' '15' '16' '19' '20' '21' '22' '23' '24' '25' '26' '27' '28' '29' '30' '31' '32' '33' '34' '35' '36')
declare -a nets=('vgg16_train-imagenet-subset' 'vgg16_train-vggface2-match-imagenet-subset')

ii=0
for sub in ${subs[@]}; do
    for net in ${nets[@]}; do
        let ii+=1
        cmd="python scripts/full_verification.py --sub-id ${sub} --comparison-net ${net} --overwrite"
        if [ ${slurm} -eq 1 ]; then
            sbatch --export="COMMAND=$cmd" --job-name sim-$ii --time 1-00:00:00 --output=log/sim-${ii}.log scripts/run_slurm.sbatch
        else
            $cmd
        fi
    done
done