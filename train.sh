# script to set variables more easily - note steps * batch_size <= training samples
python3 train.py \
    --snapshot imagenet \
    --phi 0 \
    --gpu 0 \
    --freeze-backbone \
    --weighted-bifpn \
    --compute-val-loss \
    --random-transform \
    --batch-size 64 \
    --lr 0.001 \
    --steps 10 \
    --epochs 200 \
    --hinge_loss \
    --dropout_rate 0.5 \
    csv data/train-annotations.csv data/classes.csv data/colors.csv data/bodies.csv \
    --val-annotations data/val-annotations.csv