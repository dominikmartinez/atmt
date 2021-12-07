model=bpe
part=part2
beamsize=10

for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
echo $alpha
mkdir -p assignments/04/$model/$part/$alpha
SECONDS=0

python translate_beam.py \
  --data data/en-fr/prepared \
  --dicts data/en-fr/prepared \
  --checkpoint-path assignments/03/$model/checkpoints/checkpoint_best.pt \
  --output assignments/04/$model/$part/$alpha/translations \
  --beam-size $beamsize \
  --alpha $alpha

trainingtime=$SECONDS
echo "training time: $(($trainingtime / 60)) minutes $(($trainingtime % 60)) seconds"

bash scripts/postprocess.sh assignments/04/$model/$part/$alpha/translations assignments/04/$model/$part/$alpha/postprocessed_translations en

cat assignments/04/$model/$part/$alpha/postprocessed_translations | sacrebleu data/en-fr/raw/test.en

totaltime=$SECONDS
echo "total time: $(($totaltime / 60)) minutes $(($totaltime % 60)) seconds"

done
