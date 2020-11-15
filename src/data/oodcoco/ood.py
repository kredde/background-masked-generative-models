import logging

import PIL.Image

from src.data.oodcoco import cocowrap
from src.data.oodcoco import transforms

logger = logging.getLogger(__name__)


def _filter_non_persons(image_path, annotations, min_size, rescale_size):
    """
    Yields cropped image with black background and cropped image with original background.

    For each crop, it has to hold that:

    - The instance is one single segment
    - The instance can be cropped from the image without exceeding the image dimensions
    - The cropped images shorter dimension is at least 'min_size'

    :param image_path: Path to the image.
    :param annotations: List of annotatons for the image.
    :param min_size: Minimum size of a cropped image.
    :param rescale_size: Size of the resulting images (where width = height).
    """
    image = PIL.Image.open(image_path)

    for i, anno in enumerate(annotations):
        result = transforms.filter_crops(image, anno, min_size, rescale_size)

        if result:
            cropped, cropped_original_bg = result
            yield cropped, cropped_original_bg


def generate(parameters):
    """
    :param parameters: Dict with

        parameters = {
            'names': ['dog', 'teddy bear', 'car', 'clock', 'umbrella'],

            'coco-data-dir': pathlib.Path('/data/datasets/coco2017/'),
            'annotations': pathlib.Path('annotations/person_keypoints_train2017.json'),
            'train-images': pathlib.Path('train/images'),
            'results-dir': pathlib.Path('/results'),

            'min-size': 20,
            'rescale': 96
        }
    """

    for name in parameters['names']:
        num_crops = cocowrap.iter_images(name, parameters, _filter_non_persons)
        print(f"Saved {num_crops} for category {name}.")
