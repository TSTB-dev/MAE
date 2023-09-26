#!/bin/bash

# build virtual environment
mkdir env
cd env
python -m venv mae
source mae/bin/activate
cd ..

# install pip requirements
pip install -r requirements.txt

# setup for mmdetection
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
cd ..

