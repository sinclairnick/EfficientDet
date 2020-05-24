environment: # sometimes have to run this twice
	sudo apt-get install libsm6 libxrender1 libfontconfig1
	pip3 install -r requirements.txt
	python3 setup.py build_ext --inplace

raw-data: # gdown may require sudo install
	$(shell gdown --id 1Txdl3Rjsva3ggGNZOF4y78iyTysnffh4 -O data/raw/nzvd.tar.gz) 
	$(shell gdown --id 1S6MWdY9_fk83rCHjkb6e4AJ5183pDM8W -O data/raw/stanford-cars.tar.gz)
	$(shell gdown --id 1Z-JmTJxElt3nYHYDW_cGeTRznuC8SHHp -O data/raw/car-colors.tar.gz)
	$(shell tar -C data/raw -zxvf data/raw/nzvd.tar.gz)
	$(shell tar -C data/raw -zxvf data/raw/stanford-cars.tar.gz) 
	$(shell tar -C data/raw -zxvf data/raw/car-colors.tar.gz) 

processed-data:
	python3 data/src/make-data.py

# default values for parameters
PHI=0
LR=0.0001
STEPS=10
EPOCHS=300
DROPOUT_RATE=0.5
SNAPSHOT=imagenet

BODY_DIR=data/processed/stanford-cars
COLOR_DIR=data/processed/car-colors
TRAIN_DIR=data/processed/nzvd

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
    --steps 50 \
    --epochs ${EPOCHS} \
    --dropout_rate ${DROPOUT_RATE} \
	--freeze_color \
	--wandb \
    csv ${BODY_DIR}/train_annotations.csv data/processed/classes.csv data/processed/colors.csv \
    --val-annotations ${BODY_DIR}/val_annotations.csv

pretrain-color:
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
    --steps 32 \
    --epochs ${EPOCHS} \
    --dropout_rate ${DROPOUT_RATE} \
	--freeze_body \
	--wandb \
	--no-evaluation \
    csv ${COLOR_DIR}/train_annotations.csv data/processed/classes.csv data/processed/colors.csv \
    --val-annotations ${COLOR_DIR}/val_annotations.csv

train:
	python3 train.py \
	--gpu 0 \
	--freeze-backbone \
	--weighted-bifpn \
	--compute-val-loss \
	--batch-size 32 \
	--snapshot ${SNAPSHOT} \
	--phi ${PHI} \
	--lr ${LR} \
	--steps ${STEPS} \
	--epochs ${EPOCHS} \
	--dropout_rate ${DROPOUT_RATE} \
	${FLAGS} \
	csv ${TRAIN_DIR}/train_annotations.csv \
	data/processed/classes.csv data/processed/colors.csv \
	--val-annotations ${TRAIN_DIR}/val_annotations.csv

inference:
	python3 inference.py \
	--phi ${PHI} \
	--class_path data/processed/classes.csv \
	--model_path model-best.h5 \
	--image_dir data/processed/nzvd/test \
	--colors_path data/processed/colors.csv