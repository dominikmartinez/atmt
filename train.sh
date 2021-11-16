model=dropout10

date

python train.py \
  --data data/en-fr/prepared \
  --source-lang fr \
  --target-lang en \
  --save-dir assignments/03/$model/checkpoints
#  --train-on-tiny

date
