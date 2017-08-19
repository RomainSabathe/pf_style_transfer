import numpy as np
from flask import render_template, request
from app import app, tools  # app is a Flask object
import style_transfer.main as style_transfer

# TODO: fix the extension.
LOCATION_STYLE_IMG  = 'app/static/style_img.jpg'
LOCATION_TARGET_IMG = 'app/static/target_img.jpg'
LOCATION_BLENDED_IMG = 'app/static/blend_img.jpg'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    style_img_http = request.form['style_img_http']
    target_img_http = request.form['target_img_http']

    style_img = tools.get_img(style_img_http, LOCATION_STYLE_IMG)
    target_img = tools.get_img(target_img_http, LOCATION_TARGET_IMG)
    blend_img = style_transfer.main(style_img, target_img)
    tools.save_img(blend_img, LOCATION_BLENDED_IMG)

    # TODO: hack to quickly generate a ~hash.
    hash = np.random.randint(1e9, 1e12)
    return render_template('result.html', hash=str(hash))
