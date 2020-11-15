import logging

import PIL.Image
import pycocotools.coco

from src.data.oodcoco.oodcoco import cocowrap
from src.data.oodcoco.oodcoco import plot
from src.data.oodcoco.oodcoco import transforms

logger = logging.getLogger(__name__)


def _required_keypoints_visible(keypoints, keypoint_names):
    """
    :param keypoints: (x, y, v) * num_keypoints list where v == 2 means the keypoints is visible.
    :param keypoint_names: List of keypoint names (e.g. 'nose') of length num_keypoints.

    For person we have:
    ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

    :return: True if at least one of the head keypoints and all remaining keypoints are visible.
        False otherwise.
    """
    head_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']

    # Ignore wrists.
    left_names = ['left_shoulder', 'left_elbow', 'left_hip', 'left_knee', 'left_ankle']
    right_names = ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee', 'right_ankle']
    torso_names = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']

    mapping = {}

    for i, name in enumerate(keypoint_names):
        flags = keypoints[i * 3:i * 3 + 3]
        mapping[name] = flags

    # At least one head part present.
    num_head_keypoints_visible = 0

    for name in head_names:
        if mapping[name][2] == 2:
            num_head_keypoints_visible += 1

    if num_head_keypoints_visible == 0:
        return False

    # All other should be completely visible.
    for name in left_names + right_names + torso_names:
        if mapping[name][2] != 2:
            return False

    return True


def _filter_persons(image_path, annotations, min_size, rescale_size, keypoint_names):
    """
    Iterates over all annotations of an image and yields (cropped, cropped_original_bg) where
    cropped is a cropped image of a person with width = height = 'rescale' with black background and
    cropped_original_bg is the same image but with original background.

    Images are yielded when:
    - The persons are one single segment
    - The persons are completely visible in the image
    - The persons can be cropped from the image without exceeding the image dimensions
    - The cropped images shorter dimension is at least 'min_size'

    :param image_path: Path to the image.
    :param annotations: List of annotatons for the image.
    :param min_size: Minimum size of a cropped image.
    :param rescale_size: Size of the resulting images (where width = height).
    :param keypoint_names: List of keypoint names.

    :return:
        PIL image
        mask np.array shape (height, width, 3) or (height, width) for greyvalue.
        bbox list shape (4,)
    """
    image = PIL.Image.open(image_path)

    for i, anno in enumerate(annotations):
        keypoints = anno['keypoints']

        if not _required_keypoints_visible(keypoints, keypoint_names):
            continue

        result = transforms.filter_crops(image, anno, min_size, rescale_size)

        if result:
            cropped, cropped_original_bg = result
            yield cropped, cropped_original_bg


def get_keypoint_names(parameters):
    """
    Return the keypoint names for persons category.

    :param parameters: Dict with:
        parameters = {
            'coco-data-dir': pathlib.Path(...),
            'annotations': Path to annotations file with keypoints
        }

    :return: List of keypoint names.
    """
    coco = pycocotools.coco.COCO(str(parameters['coco-data-dir'] / parameters['annotations']))
    category_id = coco.getCatIds('person')

    category_info = coco.loadCats(category_id)[0]
    keypoint_names = category_info['keypoints']

    return keypoint_names


def generate_persons(parameters):
    """
    Create a dataset of single, complete persons in various poses
    and from different points of views.
    Each resulting image is saved in two versions, one with masked black background and one with the
    original background.

    :param parameters: Dict with

        parameters = {
            'coco-data-dir': pathlib.Path('/data/datasets/coco2017/'),
            'annotations': pathlib.Path('annotations/person_keypoints_train2017.json'),
            'train-images': pathlib.Path('train/images'),
            'results-dir': pathlib.Path('/results'),

            'draw-keypoints': False,

            'min-size': 20,
            'rescale': 96
        }
    """

    def _draw_keypoints(image_name, image_path, results_dir, annotations):
        image_keypoints = plot.keypoints(image_path, annotations)
        image_keypoints.save(results_dir / str(image_name + '.png'))

    if parameters['draw-keypoints']:
        kpfun = _draw_keypoints
    else:
        kpfun = None

    name = 'person'
    keypoint_names = get_keypoint_names(parameters)

    num_crops = cocowrap.iter_images(name, parameters,
                                             lambda *args: _filter_persons(*args, keypoint_names),
                                             kpfun)

    print(f'Saved {num_crops} images.')
