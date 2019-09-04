#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -f "data/rus/train.csv" ]; then
    #echo "Downloading and preprocessing LDC93S1 example data, saving in ./data/ldc93s1."
	echo "not found: data/rus/train.csv"
    #python3 -u bin/import_ldc93s1.py ./data/ldc93s1
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python3 -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/rus"))')
fi

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python3 -u DeepSpeech.py --noshow_progressbar --noearly_stop \
  --feature_cache '/tmp/ldc93s1_cache' \
  --train_files data/rus/train.csv \
  --dev_files data/rus/dev.csv \
  --test_files data/rus/test.csv \
  --train_batch_size 1 \
  --dev_batch_size 1 \
  --test_batch_size 1 \
  --n_hidden 100 \
  --epochs 200 \
  --max_to_keep 1 
  --checkpoint_dir data/rus/checkpoint_dir \
  --learning_rate 0.001 --dropout_rate 0.05  --export_dir '/tmp/train' \
  "$@"
  
 cd /home/alex/projects/mozilla/DeepSpeech
 
 ./bin/run-rus.sh
 