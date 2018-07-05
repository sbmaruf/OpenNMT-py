#!/usr/bin/env bash
# Author : Thamme Gowda
# Created : Nov 06, 2017
#
# Running on local server 0


#======= EXPERIMENT SETUP ======
# Activate python environment if needed
#source ~/.bashrc
# source activate py3
set -e
ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# update these variables
LANG1="ms"
LANG2="en"
NAME="luong.dot.$LANG1-$LANG2"
OUT="$ONMT/onmt-runs/$NAME"
DATA="../../dataset-gen/all_dataset/alt_amara_GNOME_KDE4_OpenSubtitles2016_OpenSubtitles2018_Ubuntu"
train_src="train.en-ms.tok.clean.$LANG1"
train_tgt="train.en-ms.tok.clean.$LANG2"
dev_src="dev.en-ms.tok.$LANG1"
dev_tgt="dev.en-ms.tok.$LANG2"
test_src="test.en-ms.tok.$LANG1"
test_tgt="test.en-ms.tok.$LANG2"
TRAIN_SRC="$DATA/$train_src"
TRAIN_TGT="$DATA/$train_tgt"
VALID_SRC="$DATA/$dev_src"
VALID_TGT="$DATA/$dev_tgt"
TEST_SRC="$DATA/$test_src"
TEST_TGT="$DATA/$test_tgt"
VOCAB="" # make it empty if you want to create vocab by Open-NMT-py
VOCAB_TAG=" -src_vocab_size 60000 -tgt_vocab_size 60000 "
GPUARG="" # default
GPUARG="1"


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
cp $TRAIN_SRC $OUT/data/$train_src
cp $TRAIN_TGT $OUT/data/$train_tgt
cp $VALID_SRC $OUT/data/$dev_src
cp $VALID_TGT $OUT/data/$dev_tgt
cp $TEST_SRC $OUT/data/$test_src
cp $TEST_TGT $OUT/data/$test_tgt

#: <<EOF
# change according to your need


echo "Step 1b: Preprocess"
python $ONMT/preprocess.py \
    -train_src $OUT/data/$train_src \
    -train_tgt $OUT/data/$train_tgt \
    -valid_src $OUT/data/$dev_src \
    -valid_tgt $OUT/data/$dev_tgt \
    -save_data $OUT/data/processed \
    -share_vocab \
    $VOCAB_TAG \
    -src_seq_length 80 \
    -tgt_seq_length 80


echo "Step 2: Train"
src_word_vec_size=512
tgt_word_vec_size=512
encoder_type="brnn"
decoder_type="rnn"
rnn_size=1024
attention="dot"

GPUARG="" # default
GPUARG="0" # mention with which number of gpu you want to run the code
CMD="nohup python -u $ONMT/train.py \
        -data $OUT/data/processed \
        -save_model $OUT/models/$NAME \
        -gpuid $GPUARG \
        -batch_size 64 \
        -src_word_vec_size ${src_word_vec_size} \
        -tgt_word_vec_size ${tgt_word_vec_size} \
        -encoder_type ${encoder_type} \
        -decoder_type ${decoder_type} \
        -rnn_size ${rnn_size} \
        -global_attention ${attention} \
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
