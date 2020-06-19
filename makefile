environment: # sometimes have to run this twice
	sudo apt-get install libsm6 libxrender1 libfontconfig1
	pip3 install -r requirements.txt
	python3 setup.py build_ext --inplace

raw-data: # gdown may require sudo install
	$(shell gdown --id 1IcX2xlz2A-QostjdL_NU5SrBp_vy9bs5 -O data/raw/comp-cars.zip) 
	$(shell unzip -n data/raw/comp-cars.zip)
	$(shell mv comp-cars data/raw)

processed-data:
	python3 data/src/make_data.py

# default values for parameters
PHI=0
LR=0.0001
STEPS=25
EPOCHS=100
DROPOUT_RATE=0.5
SNAPSHOT=imagenet

BASE_DIR=data/raw/comp-cars
BODY_DIR=${BASE_DIR}/data
COLOR_DIR=${BASE_DIR}/sv_data
TRAIN_DIR=data/processed/nzvd #TODO:

pretrain-body:
	python3 train.py \
    --gpu 0 \
    --freeze-backbone \
    --weighted-bifpn \
    --compute-val-loss \
    --batch-size 32 \
	--random-transform \
    --snapshot ${SNAPSHOT} \
    --phi ${PHI} \
    --lr ${LR} \
    --steps 2500 \
    --epochs ${EPOCHS} \
    --dropout_rate ${DROPOUT_RATE} \
	--freeze_color \
	--wandb \
    csv ${BODY_DIR}/train_annotations.csv ${BODY_DIR}/classes.csv ${COLOR_DIR}/classes.csv \
    --val-annotations ${BODY_DIR}/val_annotations.csv

pretrain-color: # note: has less available arguments than above
	python3 train_color.py \
    --gpu 0 \
    --freeze-backbone \
    --weighted-bifpn \
    --batch-size 32 \
    --snapshot ${SNAPSHOT} \
    --phi ${PHI} \
    --lr ${LR} \
    --steps 32 \
    --epochs ${EPOCHS} \
    --dropout_rate ${DROPOUT_RATE} \
	--freeze_body \
	--wandb \
	--multiprocessing \
    csv ${COLOR_DIR}/train_annotations.csv ${COLOR_DIR}/classes.csv \
    --val-annotations ${COLOR_DIR}/val_annotations.csv

train:
	python3 train.py \
	--gpu 0 \
	--freeze-backbone \
	--weighted-bifpn \
	--compute-val-loss \
	--batch-size 32 \
	--random-transform \
	--snapshot ${SNAPSHOT} \
	--phi ${PHI} \
	--lr ${LR} \
	--steps ${STEPS} \
	--epochs ${EPOCHS} \
	--dropout_rate ${DROPOUT_RATE} \
	--wandb \
	csv ${TRAIN_DIR}/train_annotations.csv \
	data/processed/classes.csv data/processed/colors.csv \
	--val-annotations ${TRAIN_DIR}/val_annotations.csv

IMAGE_DIR=data/processed/nzvd/train
MODEL_PATH=weights/extracted-weights-phi0.h5
inference:
	python3 inference.py \
	--phi ${PHI} \
	--model_path ${MODEL_PATH} \
	--class_path data/processed/classes.csv \
	--image_dir ${IMAGE_DIR} \
	--colors_path data/processed/colors.csv

merge-weights:
	python3 merge_weights.py \
	--color_weights color_weights.h5 \
	--body_weights body_weights.h5 \
	--phi ${PHI} \
	--class_path data/processed/classes.csv \
	--colors_path data/processed/colors.csv \

PREDICTIONS_PATH=predictions.csv
evaluate:
	python3 evaluate.py \
	--annotations_path ${IMAGE_DIR}_annotations.csv \
	--predictions_path  ${PREDICTIONS_PATH} \
	--classes_path data/processed/classes.csv \
	--colors_path data/processed/colors.csv

similarities:
	python3 similarities.py 