import logging
import pathlib

import oodcoco.id
import oodcoco.ood

import oodcoco.cocowrap

logger = logging.getLogger(__name__)


def run():
    logging.basicConfig(level=logging.WARNING)

    parameters = {
        'coco-data-dir': pathlib.Path('/data/datasets/coco2017/'),
        'annotations': pathlib.Path('annotations/person_keypoints_train2017.json'),
        'train-images': pathlib.Path('train/images'),
        'results-dir': pathlib.Path('results'),

        'min-size': 20,
        'rescale': 96
    }

    oodcoco.id.generate_persons(parameters)

    parameters = {
        'names': ['dog', 'teddy bear', 'car', 'clock', 'umbrella'],

        'coco-data-dir': pathlib.Path('/data/datasets/coco2017/'),
        'annotations': pathlib.Path('annotations/instances_train2017.json'),
        'train-images': pathlib.Path('train/images'),
        'results-dir': pathlib.Path('results'),

        'min-size': 20,
        'rescale': 96
    }

    oodcoco.ood.generate(parameters)


if __name__ == '__main__':
    run()
