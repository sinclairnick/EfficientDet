input_dir = 'data/raw'
output_dir = 'data/processed'

TLHW = ['top', 'left', 'height', 'width']
XYXY = ['x1', 'y1', 'x2', 'y2']
OUT_HEADER = ['file', *XYXY, 'body', 'color']

mappings = {
    'supercab': 'ute',
    'cab': 'ute',
    'minivan': 'people-mover',
    'wagon': 'station-wagon',
}