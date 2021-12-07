model=bpe
part=part1

for beamsize in {1..12}
do

echo $beamsize
mkdir -p assignments/04/$model/$part/$beamsize
SECONDS=0

python translate_beam.py \
  --data data/en-fr/prepared \
  --dicts data/en-fr/prepared \
  --checkpoint-path assignments/03/$model/checkpoints/checkpoint_best.pt \
  --output assignments/04/$model/$part/$beamsize/translations \
  --beam-size $beamsize

trainingtime=$SECONDS
echo "training time: $(($trainingtime / 60)) minutes $(($trainingtime % 60)) seconds"

bash scripts/postprocess.sh assignments/04/$model/$part/$beamsize/translations assignments/04/$model/$part/$beamsize/postprocessed_translations en

cat assignments/04/$model/$part/$beamsize/postprocessed_translations | sacrebleu data/en-fr/raw/test.en

totaltime=$SECONDS
echo "total time: $(($totaltime / 60)) minutes $(($totaltime % 60)) seconds"

done
