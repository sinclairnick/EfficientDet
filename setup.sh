pip3 install gdown
python3 setup.py build_ext --inplace
pip3 install -r requirements.txt
gdown --id 1MNB5q6rJ4TK_gen3iriu8-ArG9jB8aR9 -O coco_weights/efficientdet-d0.h5 # coco  phi=0
gdown --id 11pQznCTi4MaVXqkJmCMcQhphMXurpx5Z -O coco_weights/efficientdet-d1.h5 # coco phi=1
gdown --id 1_yXrOrY0FDnH-d_FQIPbGy4z2ax4aNh8 -O coco_weights/efficientdet-d2.h5 # coco phi=2