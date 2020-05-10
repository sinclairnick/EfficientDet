# script to set variables more easily - note steps * batch_size <= training samples
python3 train.py \
    --snapshot imagenet \
    --phi 0 \
    --gpu 0 \
    --weighted-bifpn \
    --random-transform \
    --compute-val-loss \
    --freeze-backbone \
    --batch-size 70 \
    --lr 0.001 \
    --steps 10 \
    --epochs 200 \
    --hinge_loss \
    --dropout_rate 0.1 \
    --wandb \
    csv data/train-cropped-rect.csv data/classes.csv data/colors.csv data/bodies.csv \
    --val-annotations data/val-cropped-rect.csv