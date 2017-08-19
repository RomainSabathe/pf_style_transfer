import numpy as np
import requests
import logging
from PIL import Image


def get_img(img_http, dest=None):
    """Downloads an image from the web and returns it as a numpy array.
    The image can be stored if a `dest` is given.

    Args:
        img_http (str): a valid html address pointing to an image.
        dest (str): where the image will be stored locally.
    """
    logging.debug('Making a request to get image.')
    req = requests.get(img_http)

    logging.debug('Saving image to dest.')
    # TODO: check image extension.
    # TODO: handle dest check... (existence)
    dest = dest if dest is not None else 'tmp/tmp_img.jpg'
    with open(dest, 'w') as f:
        f.write(req.content)
    return np.array(Image.open(dest))
