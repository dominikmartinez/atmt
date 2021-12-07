model=bpe
part=part4
beamsize=10
alpha=0.3
n_hyp=10

mkdir -p assignments/04/$model/$part
SECONDS=0

python translate_beam.py \
  --data data/en-fr/prepared \
  --dicts data/en-fr/prepared \
  --checkpoint-path assignments/03/$model/checkpoints/checkpoint_best.pt \
  --output assignments/04/$model/$part/translations \
  --beam-size $beamsize \
  --alpha $alpha \
  --n-hyp $n_hyp

trainingtime=$SECONDS
echo "training time: $(($trainingtime / 60)) minutes $(($trainingtime % 60)) seconds"

bash scripts/postprocess.sh assignments/04/$model/$part/translations assignments/04/$model/$part/postprocessed_translations en

cat assignments/04/$model/$part/postprocessed_translations | sacrebleu data/en-fr/raw/test.en

totaltime=$SECONDS
echo "total time: $(($totaltime / 60)) minutes $(($totaltime % 60)) seconds"
