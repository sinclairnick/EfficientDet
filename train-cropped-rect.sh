# script to set variables more easily - note steps * batch_size <= training samples
python3 train.py \
    --snapshot imagenet \
    --phi 0 \
    --gpu 0 \
    --freeze-backbone \
    --weighted-bifpn \
    --compute-val-loss \
    --batch-size 35 \
    --lr 0.0001 \
    --steps 10 \
    --epochs 200 \
    --dropout_rate 0.5 \
    --wandb \
    csv data/train-cropped-rect.csv data/classes.csv data/colors.csv data/bodies.csv \
    --val-annotations data/val-cropped-rect.csv