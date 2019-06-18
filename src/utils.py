def crop_images(position, shape, *args):
    assert len(args) > 0, "At least 1 image needed."

    x, y = position
    w, h = shape
    cropped = []

    for img in args:
        cropped.append(img[x : x + h, y : y + w])

    return cropped
