# script to set variables more easily - note steps * batch_size <= training samples
python3 train.py \
    --detect-quadrangle \
    --snapshot imagenet \
    --phi 0 \
    --gpu 0 \
    --random-transform \
    --compute-val-loss \
    --freeze-backbone \
    --batch-size 32 \
    --steps 10 \
    --epochs 100 \
    --hinge_loss \
    --dropout_rate 0.1 \
    --wandb \
    csv data/train.csv data/classes.csv data/colors.csv data/bodies.csv \
    --val-annotations data/val.csv