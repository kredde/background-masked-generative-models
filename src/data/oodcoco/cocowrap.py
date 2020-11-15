import numpy as np

import pycocotools.coco

from src.data.oodcoco import plot


def get_category_info(category_names, coco):
    """
    Get the coco category info and image ids for given category names.

    :param category_names: List of category names.
    :param coco: Pycocotools COCO instance.

    :return: Dict with:
        category_info['person'] = (id, [image_ids, ...])
    """
    category_ids = zip(category_names, coco.getCatIds(catNms=category_names))

    info = {}

    for category_name, category_id in category_ids:
        image_ids = coco.getImgIds(catIds=category_id)

        info[category_name] = (category_id, image_ids)

    return info


def iter_images(category_name, parameters, _filter, image_callback=None):
    """
    :param category_name: COCO category name.
    :param parameters: Dict with keys:
        parameters = {
            'coco-data-dir': pathlib.Path('/data/datasets/coco2017/'),
            'annotations': pathlib.Path('annotations/person_keypoints_train2017.json'),
            'train-images': pathlib.Path('train/images'),
            'results-dir': pathlib.Path('results')
        }
    :param _filter: Callable with _filter(image_path, annotations, min_size, rescale).
    :param image_callback: If not None, is called for each image with arguments
        image_callback(image_name, image_path, results_dir, annotations)
    """
    coco = pycocotools.coco.COCO(str(parameters['coco-data-dir'] / parameters['annotations']))

    category_id, image_ids = get_category_info([category_name], coco)[category_name]

    results_dir = parameters['results-dir'] / category_name
    results_dir.mkdir(parents=True, exist_ok=True)

    num_crops = 0
    plot_images = []

    for i, image_id in enumerate(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=category_id, iscrowd=0))

        image_path = parameters['coco-data-dir'] / parameters['train-images'] / (
            image_info['file_name'])

        image_name = f'{category_name}-{image_id}-{i}'

        if image_callback:
            image_callback(image_name, image_path, results_dir, annotations)

        for j, (cropped, cropped_original_bg) in enumerate(
                _filter(image_path,
                        annotations,
                        parameters['min-size'],
                        parameters['rescale'])):

            num_crops += 1

            crop_name = image_name + f'-{j}'

            cropped.save(results_dir / str(crop_name + '.png'))
            cropped_original_bg.save(results_dir / str(crop_name + '-bg.png'))

            if len(plot_images) < 36:
                plot_images.append(np.array(cropped))
                plot_images.append(np.array(cropped_original_bg))

                if len(plot_images) == 36:
                    plot.grid(np.array(plot_images), 6, 6, results_dir / 'examples.pdf')

    return num_crops
