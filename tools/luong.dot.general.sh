#!/usr/bin/env bash
# Author : Thamme Gowda
# Created : Nov 06, 2017
#
# Running on local server 0

ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

#======= EXPERIMENT SETUP ======
# Activate python environment if needed
source ~/.bashrc
# source activate py3

# update these variables
NAME="luong.dot"
OUT="onmt-runs/$NAME"

DATA="../dataset-gen/all_dataset/alt_amara_GNOME_KDE4_OpenSubtitles2016_OpenSubtitles2018_Ubuntu"
TRAIN_SRC=$DATA/train.en-ms.tok.en
TRAIN_TGT=$DATA/train.en-ms.tok.ms
VALID_SRC=$DATA/dev.en-ms.tok.en
VALID_TGT=$DATA/dev.en-ms.tok.ms
TEST_SRC=$DATA/test.en-ms.tok.en
TEST_TGT=$DATA/test.en-ms.tok.ms

BPE="" # default
# BPE="src+tgt" # src, tgt, src+tgt

# applicable only when BPE="src" or "src+tgt"
BPE_SRC_OPS=50000

# applicable only when BPE="tgt" or "src+tgt"
BPE_TGT_OPS=50000


#====== EXPERIMENT BEGIN ======

# Check if input exists
for f in $TRAIN_SRC $TRAIN_TGT $VALID_SRC $VALID_TGT $TEST_SRC $TEST_TGT; do
    if [[ ! -f "$f" ]]; then
        echo "Input File $f doesnt exist. Please fix the paths"
        exit 1
    fi
done

function lines_check {
    l1=`wc -l $1`
    l2=`wc -l $2`
    IFS=' ' read -a cnt1 <<< l1;
    IFS=' ' read -a cnt2 <<< l1;
    if [[ ${cnt1[0]} != ${cnt2[0]} ]]; then
        echo "Number of lines in $1 is $l1"
        echo "Number of lines in $2 is $l2"
        echo "ERROR: Record counts doesnt match between: $1 and $2"
        exit 2
    fi
}
lines_check $TRAIN_SRC $TRAIN_TGT
lines_check $VALID_SRC $VALID_TGT
lines_check $TEST_SRC $TEST_TGT

#set the output directory
echo "Output dir = $OUT"
[ -d $OUT ] || mkdir -p $OUT
[ -d $OUT/data ] || mkdir -p $OUT/data
[ -d $OUT/models ] || mkdir $OUT/models
[ -d $OUT/test ] || mkdir -p  $OUT/test


echo "Step 1a: Preprocess inputs"
if [[ "$BPE" == *"src"* ]]; then
    echo "BPE on source"
    # Here we could use more  monolingual data
    $ONMT/tools/learn_bpe.py -s $BPE_SRC_OPS < $TRAIN_SRC > $OUT/data/bpe-codes.src

    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $TRAIN_SRC > $OUT/data/train.src
    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $VALID_SRC > $OUT/data/valid.src
    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $TEST_SRC > $OUT/data/test.src
else
    cp $TRAIN_SRC $OUT/data/train.src
    cp $VALID_SRC $OUT/data/valid.src
    cp $TEST_SRC $OUT/data/test.src
fi


if [[ "$BPE" == *"tgt"* ]]; then
    echo "BPE on target"
    # Here we could use more  monolingual data
    $ONMT/tools/learn_bpe.py -s $BPE_SRC_OPS < $TRAIN_TGT > $OUT/data/bpe-codes.tgt

    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.tgt <  $TRAIN_TGT > $OUT/data/train.tgt
    $ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.tgt <  $VALID_TGT > $OUT/data/valid.tgt
    #$ONMT/tools/apply_bpe.py -c $OUT/data/bpe-codes.tgt <  $TEST_TGT > $OUT/data/test.tgt
    # We dont touch the test References, No BPE on them!
    cp $TEST_TGT $OUT/data/test.tgt
else
    cp $TRAIN_TGT $OUT/data/train.tgt
    cp $VALID_TGT $OUT/data/valid.tgt
    cp $TEST_TGT $OUT/data/test.tgt
fi


#: <<EOF
# change according to your need


echo "Step 1b: Preprocess"
python $ONMT/preprocess.py \
    -train_src $OUT/data/train.src \
    -train_tgt $OUT/data/train.tgt \
    -valid_src $OUT/data/valid.src \
    -valid_tgt $OUT/data/valid.tgt \
    -save_data $OUT/data/processed


echo "Step 2: Train"
src_word_vec_size=512
tgt_word_vec_size=512
encoder_type="brnn"
decoder_type="rnn"
rnn_size=1024
global_attention="general"

GPUARG="" # default
GPUARG="0" # mention with which number of gpu you want to run the code
CMD="nohup python -u $ONMT/train.py \
        -data $OUT/data/processed \
        -save_model $OUT/models/$NAME \
        -gpuid $GPUARG \
        -src_word_vec_size ${src_word_vec_size} \
        -tgt_word_vec_size ${tgt_word_vec_size} \
        -encoder_type ${encoder_type} \
        -decoder_type ${decoder_type} \
        -rnn_size ${rnn_size} \
        -global_attention ${global_attention} \
        -epochs 100  &> $OUT/run.log &"

echo "Training command :: $CMD"
eval "$CMD"

##EOF
#
## select a model with high accuracy and low perplexity
## TODO: currently using linear scale, maybe not be the best
#model=`ls $OUT/models/*.pt| awk -F '_' 'BEGIN{maxv=-1000000} {score=$(NF-3)-$(NF-1); if (score > maxv) {maxv=score; max=$0}}  END{ print max}'`
#echo "Chosen Model = $model"
#if [[ -z "$model" ]]; then
#    echo "Model not found. Looked in $OUT/models/"
#    exit 1
#fi
#
#GPU_OPTS=""
#if [ ! -z $GPUARG ]; then
#    GPU_OPTS="-gpu $GPUARG"
#fi
#
#echo "Step 3a: Translate Test"
#python $ONMT/translate.py -model $model \
#    -src $OUT/data/test.src \
#    -output $OUT/test/test.out \
#    -replace_unk  -verbose $GPU_OPTS > $OUT/test/test.log
#
#echo "Step 3b: Translate Dev"
#python $ONMT/translate.py -model $model \
#    -src $OUT/data/valid.src \
#    -output $OUT/test/valid.out \
#    -replace_unk -verbose $GPU_OPTS > $OUT/test/valid.log
#
#if [[ "$BPE" == *"tgt"* ]]; then
#    echo "BPE decoding/detokenising target to match with references"
#    mv $OUT/test/test.out{,.bpe}
#    mv $OUT/test/valid.out{,.bpe}
#    cat $OUT/test/valid.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid.out
#    cat $OUT/test/test.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test.out
#fi
#
#echo "Step 4a: Evaluate Test"
#$ONMT/tools/multi-bleu-detok.perl $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.tc.bleu
#$ONMT/tools/multi-bleu-detok.perl -lc $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.lc.bleu
#
#echo "Step 4b: Evaluate Dev"
#$ONMT/tools/multi-bleu-detok.perl $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.tc.bleu
#$ONMT/tools/multi-bleu-detok.perl -lc $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.lc.bleu

#===== EXPERIMENT END ======
