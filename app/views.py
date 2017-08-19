from flask import render_template, request
from app import app, tools  # app is a Flask object

# TODO: fix the extension.
LOCATION_STYLE_IMG  = 'tmp/style_img.jpg'
LOCATION_TARGET_IMG = 'tmp/target_img.jpg'

@app.route('/')
def index():
    return render_template(
        'index.html',
        title='PF-ST')

@app.route('/process', methods=['POST'])
def process():
    style_img_http = request.form['style_img_http']
    target_img_http = request.form['target_img_http']

    # FOR DEBUGGING
    style_img_http = u'https://mollymusic13.files.wordpress.com/' \
                      '2015/03/meddle.jpg'
    target_img_http = u'https://upload.wikimedia.org/wikipedia/' \
                       'en/9/93/Tame_Impala_Innerspeaker_cover.jpg'

    style_img = tools.get_img(style_img_http, LOCATION_STYLE_IMG)
    target_img = tools.get_img(target_img_http, LOCATION_TARGET_IMG)
    return render_template(
        'success.html',
        title='PF-ST')
