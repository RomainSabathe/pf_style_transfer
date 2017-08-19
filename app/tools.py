import os
import shutil
import requests
import logging
from PIL import Image
from time import gmtime, strftime


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
    if dest is not None:
        create_backup(dest)
    else:
        dest = '/tmp/tmp_img.jpg'
    with open(dest, 'w') as f:
        f.write(req.content)
    return Image.open(dest)


def save_img(pil_img, dest):
    """Saves  PIL image to disk.

    Args:
        pil_img (PIL.Image): the image to be saved.
        dest (str): where the image should be saved.
    """
    create_backup(dest)
    pil_img.save(dest)


def create_backup(dest):
    """If a file exists at `dest`, we create a copy of it by associating it
    with a unique timestamp."""
    if not os.path.exists(dest):
        return
    base_name = os.path.basename(dest)
    folder_name = os.path.dirname(dest)
    timestamp = strftime('%Y%m%d_%H%M%S', gmtime())
    new_dest = os.path.join(folder_name, '{}_{}'.format(timestamp, base_name))
    shutil.copy(dest, new_dest)
