def crop_images(x, y, w, h, *args):
    """
    Crops all the images passed as parameter using the box coordinates passed
    """
    assert len(args) > 0, "At least 1 image needed."
    
    cropped = []
    for img in args:
        cropped.append(img[x : x + h, y : y + w])

    return cropped
