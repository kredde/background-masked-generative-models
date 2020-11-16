import pathlib
import PIL.Image
import PIL.ImageDraw
import matplotlib.pyplot as plt

matplotlib_params = {
    'backend': 'pgf',
    'pgf.texsystem': 'lualatex',
    'text.latex.preamble': [r"\usepackage{lmodern}"],
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 6,
    'font.family': 'serif',
}

plt.rcParams.update(matplotlib_params)


def keypoints(image_path, annotations):
    """
    Draw the keypoints onto an image.
    Visible keypoints are red, occluded keypoints are blue.

    :param image_path: Path to the image.
    :param annotations: Pycocotools annotations.

    :return: PIL image with drawn keypoints for all persons.
    """
    image = PIL.Image.open(image_path)
    draw = PIL.ImageDraw.Draw(image)

    for i, anno in enumerate(annotations):
        kp = anno['keypoints']

        for j in range(0, len(kp), 3):
            _x, _y, _v = kp[j:j + 3]

            if _v == 2:
                draw.ellipse([(_x - 1, _y - 1), (_x + 1, _y + 1)], fill='red')
            elif _v == 1:
                draw.ellipse([(_x - 1, _y - 1), (_x + 1, _y + 1)], fill='blue')

    return image


def grid(images, rows, cols, filepath=None):
    """
    :param images: Numpy array shape (N, height, width, channels).
    :param rows: Number of rows of the plot.
    :param cols: Number of columns of the plot.
    :param filepath: If not None, the plot is saved as 'filepath'. The filepath should include the
        filetype (e.g. plot.pdf).
    """
    if images.shape[0] != rows * cols:
        raise ValueError("Number of images is not equal to rows * cols.")

    filepath = pathlib.Path(filepath)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(32, 32))
    axes = axes.flatten()

    for img, ax in zip(images, axes):
        ax.imshow(img.squeeze())
        ax.axis('off')

    plt.tight_layout()

    if filepath:
        filepath.parents[0].mkdir(parents=True, exist_ok=True)
        plt.savefig(str(filepath))

        print(f"Saved image as {filepath}.")
    else:
        plt.show()
