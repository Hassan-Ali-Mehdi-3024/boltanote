#!/usr/bin/env bash
# Downloads the three public datasets into $1 (default: data_raw), laid out as boltanote/data.py expects.
set -e
D=${1:-data_raw}; mkdir -p "$D"; cd "$D"
if [ ! -d rafay ]; then
  curl -sL -o rafay.zip https://www.kaggle.com/api/v1/datasets/download/rafayqayyum/pakistan-image-currency-dataset
  mkdir -p tmp_r && unzip -q rafay.zip -d tmp_r && mv tmp_r/currencyDataset rafay && rm -rf tmp_r rafay.zip
fi
if [ ! -d imagenette ]; then
  curl -sL https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz -o inet.tgz
  mkdir -p imagenette && tar -xzf inet.tgz -C imagenette && rm inet.tgz
fi
if [ ! -d roboflow ]; then
  curl -sL -o rf.zip https://www.kaggle.com/api/v1/datasets/download/muhammadasjadali/pakistani-currency-notes
  mkdir -p tmp_f && unzip -q rf.zip -d tmp_f && mv "tmp_f/Pakistani Currency Notes Dataset.v1-2kbase.yolov11" roboflow && rm -rf tmp_f rf.zip
fi
echo "rafay: $(find rafay -type f | wc -l)  imagenette: $(find imagenette -name '*.JPEG' | wc -l)  roboflow: $(find roboflow -name '*.jpg' | wc -l)"
