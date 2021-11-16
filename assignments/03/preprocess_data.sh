#!/bin/bash
# -*- coding: utf-8 -*-

set -e

#pwd=`dirname "$(readlink -f "$0")"`
#base=$pwd/../..
#base=./../..
base=/home/demeter/programming/atmt_assignments/atmt
src=fr
tgt=en
data=$base/data/$tgt-$src
bpe_operations=2000

# change into base directory to ensure paths are valid
cd $base

# create preprocessed directory
mkdir -p $data/preprocessed/

# normalize and tokenize raw data
cat $data/raw/train.$src | perl $base/moses_scripts/normalize-punctuation.perl -l $src | perl $base/moses_scripts/tokenizer.perl -l $src -a -q > $data/preprocessed/train.$src.p
cat $data/raw/train.$tgt | perl $base/moses_scripts/normalize-punctuation.perl -l $tgt | perl $base/moses_scripts/tokenizer.perl -l $tgt -a -q > $data/preprocessed/train.$tgt.p

# train truecase models
perl $base/moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$src --corpus $data/preprocessed/train.$src.p
perl $base/moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$tgt --corpus $data/preprocessed/train.$tgt.p

# apply truecase models to splits
cat $data/preprocessed/train.$src.p | perl $base/moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/train.$src 
cat $data/preprocessed/train.$tgt.p | perl $base/moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/train.$tgt

# prepare remaining splits with learned models
for split in valid test tiny_train
do
    cat $data/raw/$split.$src | perl $base/moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/$split.$src
    cat $data/raw/$split.$tgt | perl $base/moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/$split.$tgt
done

# learn BPE model
subword-nmt learn-joint-bpe-and-vocab --input $data/preprocessed/train.$src $data/preprocessed/train.$tgt -s $bpe_operations -o $data/preprocessed/bpe.codes --write-vocabulary $data/preprocessed/vocab.$src $data/preprocessed/vocab.$tgt

#p apply BPE to all splits
for split in train valid test tiny_train
    do for lang in $src $tgt
        do subword-nmt apply-bpe -c $data/preprocessed/bpe.codes --vocabulary $data/preprocessed/vocab.$lang --vocabulary-threshold 5 < $data/preprocessed/$split.$lang > $data/preprocessed/$split.bpe.$lang
        rm $data/preprocessed/$split.$lang
        mv $data/preprocessed/$split.bpe.$lang $data/preprocessed/$split.$lang
    done
done

# remove tmp files
rm $data/preprocessed/train.$src.p
rm $data/preprocessed/train.$tgt.p

# preprocess all files for model training
#python preprocess.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared/ --train-prefix $data/preprocessed/train --valid-prefix $data/preprocessed/valid --test-prefix $data/preprocessed/test --tiny-train-prefix $data/preprocessed/tiny_train --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000 --vocab-src $data/preprocessed/vocab.$src --vocab-trg $data/preprocessed/vocab.$tgt
python preprocess.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared/ --train-prefix $data/preprocessed/train --valid-prefix $data/preprocessed/valid --test-prefix $data/preprocessed/test --tiny-train-prefix $data/preprocessed/tiny_train --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000

echo "done!"
