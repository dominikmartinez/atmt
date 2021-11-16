model=dropout10

python translate.py \
  --data data/en-fr/prepared \
  --dicts data/en-fr/prepared \
  --checkpoint-path assignments/03/$model/checkpoints/checkpoint_best.pt \
  --output assignments/03/$model/translations

bash scripts/postprocess.sh assignments/03/$model/translations assignments/03/$model/postprocessed_translations en

cat assignments/03/$model/postprocessed_translations | sacrebleu data/en-fr/raw/test.en
