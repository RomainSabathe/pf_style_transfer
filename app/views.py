import os
import numpy as np
from flask import render_template, request
from app import app, tools  # app is a Flask object
import style_transfer.main as style_transfer

# TODO: fix the extension.
LOCATION_PF_IMAGES = 'app/static/pinkfloyd'
LOCATION_STYLE_IMG  = 'app/static/style_img.jpg'
LOCATION_TARGET_IMG = 'app/static/target_img.jpg'
LOCATION_BLENDED_IMG = 'app/static/blend_img.jpg'

@app.route('/')
def index():
    pf_img_names = os.listdir(LOCATION_PF_IMAGES)
    # TODO: a hack to get it working with Flask's `url_for` mechanism.
    pf_img_names = [
        os.path.join('pinkfloyd', img_name) for img_name in pf_img_names]
    return render_template('index.html', pf_img_names=pf_img_names)

@app.route('/process', methods=['POST'])
def process():
    # Load the Pink Floyd image.
    # TODO: hack to quickly get access to the image.
    pf_img = tools.open_img(
        os.path.join('app/static', request.form['pf_img_choice']))

    # Load the target image.
    target_img_http = request.form['target_img_http']
    target_img = tools.get_img(target_img_http, LOCATION_TARGET_IMG)

    # Blending.
    if request.form['pf_role'] == 'style':
        # We want to extract the style of the Pink Floyd image.
        args = {'style_img': pf_img, 'content_img': target_img}
        tools.save_img(pf_img, LOCATION_STYLE_IMG)
    else:
        # We want to extract the content of the Pink Floyd image.
        args = {'style_img': target_img, 'content_img': pf_img}
        tools.save_img(pf_img, LOCATION_TARGET_IMG)
    blend_img = style_transfer.main(**args)
    tools.save_img(blend_img, LOCATION_BLENDED_IMG)

    # TODO: hack to quickly generate a ~hash.
    hash = np.random.randint(1e9, 1e12)
    return render_template('result.html', hash=str(hash))
