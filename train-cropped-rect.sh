# script to set variables more easily - note steps * batch_size <= training samples
python3 train.py \
    --snapshot imagenet \
    --phi 2 \
    --gpu 0 \
    --weighted-bifpn \
    --compute-val-loss \
    --batch-size 70 \
    --lr 0.0001 \
    --steps 10 \
    --epochs 200 \
    --dropout_rate 0.3 \
    --wandb \
    csv data/train-cropped-rect.csv data/classes.csv data/colors.csv data/bodies.csv \
    --val-annotations data/val-cropped-rect.csv