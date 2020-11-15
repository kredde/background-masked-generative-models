import math
import logging

import PIL.Image
import numpy as np
import scipy.signal

import pycocotools.mask

logger = logging.getLogger(__name__)


def filter_crops(image, anno, min_size, rescale_size):
    x, y, width, height = anno['bbox']
    segmentation = anno['segmentation']

    x, y = int(x), int(y)
    width, height = math.ceil(width), math.ceil(height)

    mask = pycocotools.mask.frPyObjects(segmentation, image.height, image.width)
    decoded_mask = np.array(pycocotools.mask.decode(mask))

    # We do not want parts of categories.
    if decoded_mask.shape[2] > 1:
        logger.info(f"Ignore crop from image due to multiple parts.")
        return False

    bbox = rescale(x, y, width, height)

    # Check if bb is inside the image.
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > image.width or bbox[3] > image.height:
        logger.info(f"Ignore crop from image due to non valid bounding box.")
        return False

    cropped, cropped_original_bg = crop(image, decoded_mask.squeeze(), bbox)

    # Check if bb size is at least min_size.
    if min(cropped.height, cropped.width) < min_size:
        logger.info(f"Ignore crop from image due to small size.")
        return False

    # There are images in the dataset that are not correctly cropped and contain large black
    # borders. Using them would result in images with incomplete background so we ignore them.
    # Natural images that have large amount of completely black areas should be rare.
    kernel_size = int((min(cropped_original_bg.height, cropped_original_bg.width) / 100) * 20)

    if has_border(cropped_original_bg, kernel_size=kernel_size):
        logger.info(f"Ignore crop from image due to large amount of black.")
        return False

    cropped = cropped.resize((rescale_size,) * 2)
    cropped_original_bg = cropped_original_bg.resize((rescale_size,) * 2)

    return cropped, cropped_original_bg


def has_border(image, kernel_size=10):
    image = np.array(image)

    if len(image.shape) == 2:
        # Greyscale image.
        kernel = np.ones((kernel_size, kernel_size))
    else:
        # Color image.
        kernel = np.ones((kernel_size, kernel_size, 3))

    filtered = scipy.signal.convolve(image, kernel, mode='valid')

    if (filtered <= 0).sum() != 0:
        return True

    return False


def rescale(x, y, width, height, padding=10):
    """
    Rescale the bounding box to have the same width and height and add some padding.
    The rescaling is done by increasing the size of the shorter side of the bounding box.

    :param padding: Number of pixels to add to each side of the bounding box.

    :return: [x1, y1, x2, y2]
    """
    bbox = np.array([[x, y],
                     [x + width, y + height]], dtype=np.float)

    maxdim = np.array([height, width]).argmax()

    delta = abs((height - width)) / 2
    delta_vec = np.array([-np.floor(delta), np.ceil(delta)], dtype=np.float)

    bbox[:, maxdim] += delta_vec

    bbox[0, :] -= padding
    bbox[1, :] += padding

    bbox_flat = bbox.flatten().tolist()

    return bbox_flat


def crop(image, mask, bbox):
    """
    Crops bbox from image and applies mask to set the background to black.
    Additionally returns a cropped image where the mask was not applied.

    :param image: PIL image.
    :param mask: numpy array (height, witdth, 3) or (height, width).
    :param bbox: List with [x, y, width, height].

    :return:
        PIL image with cropped bbox and masked background
        PIL image with cropped bbox and original background
    """
    masked = np.array(image)

    if len(masked.shape) == 2:
        # Greyscale image.
        background_color = 0
    else:
        # Color image.
        background_color = (0, 0, 0)

    masked[mask[:, :] == 0] = background_color

    cropped = PIL.Image.fromarray(masked).crop(bbox)
    cropped_original_bg = image.crop(bbox)

    return cropped, cropped_original_bg
