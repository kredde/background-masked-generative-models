This code extracts a subset of the [COCO](https://cocodataset.org/#home) dataset
for out-of-distribution (OOD) detection.

The new dataset is intended to be similar to the MNIST or Fashion-MNIST dataset.
The in-distribution (ID) dataset consist of cropped images of persons.
The OOD dataset consists of images of dogs, cars, umbrellas, clocks and teddy bears.

- Crops are rescaled to have the same dimensions (width = height = 96px)
- Crops that contain black image borders are filtered out.
- Crops that are not completely inside the original image are filtered out.
- Crops that are too small are filtered out.

Additionally, for the ID dataset the keypoint information is used to ensure a person is completely visible in an image.
This prevents samples of the ID dataset being occluded or partly visible.

For each crop, an image with masked black background and an image with the original background is available.

To run this code:

```
make
# Adapt volume binds and group ids.
vim run-docker.sh
./run-docker.sh
```

![Persons](examples/person-examples.pdf)
![Dogs](examples/dog-examples.pdf)
![Clocks](examples/clock-examples.pdf)
