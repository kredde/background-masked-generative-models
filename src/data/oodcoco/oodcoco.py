import logging
import pathlib

from src.data.oodcoco.id import generate_persons
from src.data.oodcoco.ood import generate

import src.data.oodcoco.cocowrap

logger = logging.getLogger(__name__)


def run():
    logging.basicConfig(level=logging.WARNING)

    parameters = {
        'coco-data-dir': pathlib.Path('./data/COCO/'),
        'annotations': pathlib.Path('annotations/person_keypoints_train2017.json'),
        'train-images': pathlib.Path('train2017/'),
        'results-dir': pathlib.Path('./data/COCO/foreground_images'),
        'draw-keypoints': False,

        'min-size': 20,
        'rescale': 96
    }

    generate_persons(parameters)

    parameters = {
        'names': ['dog', 'teddy bear', 'car', 'clock', 'umbrella'],

        'coco-data-dir': pathlib.Path('./data/COCO/'),
        'annotations': pathlib.Path('annotations/instances_train2017.json'),
        'train-images': pathlib.Path('train2017/'),
        'results-dir': pathlib.Path('./data/COCO/foreground_images'),
        'draw-keypoints': False,

        'min-size': 20,
        'rescale': 96
    }

    generate(parameters)


# if __name__ == '__main__':
#     run()
