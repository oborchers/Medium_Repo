#!/bin/bash
mkdir universal-sentence-encoder-5
cd universal-sentence-encoder-5
wget https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder-large/5.tar.gz
tar -xvzf 5.tar.gz
rm 5.tar.gz
cd ..
python -m tf2onnx.convert --saved-model universal-sentence-encoder-5 --output encoder/universal-sentence-encoder-5.onnx --opset 12 --extra_opset ai.onnx.contrib:1 --tag serve