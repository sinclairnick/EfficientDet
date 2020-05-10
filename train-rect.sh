# script to set variables more easily - note steps * batch_size <= training samples
python3 train.py \
    --snapshot imagenet \
    --phi 4 \
    --gpu 0 \
    --weighted-bifpn \
    --random-transform \
    --compute-val-loss \
    --freeze-backbone \
    --batch-size 35 \
    --lr 0.001 \
    --steps 10 \
    --epochs 100 \
    --hinge_loss \
    --dropout_rate 0.1 \
    --wandb \
    csv data/train-rect.csv data/classes.csv data/colors.csv data/bodies.csv \
    --val-annotations data/val-rect.csv