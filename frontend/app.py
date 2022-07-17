from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from base64 import b64encode
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import sys
sys.path.append(os.path.abspath('../models'))
from cnn_predict_visualize import load_model, load_image_from_bytes, predict, grad_cam, show_heatmap

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')
    
@app.route("/predict", methods=['GET', 'POST'])
def upload():
    global MODEL, GRAPH
    if request.method == 'POST':
        if 'image_file' not in request.files:
            flash('未上传图片或图片无效，请重新上传')
            return redirect(url_for('index'))
        f = request.files['image_file']
        if f.filename == '':
            flash('未上传图片，请重新上传')
            return redirect(url_for('index'))
        elif not allowed_file(f.filename):
            flash('请上传扩展名为png、jpg、jpeg的图片')
            return redirect(url_for('index'))
        # f.save(secure_filename(f.filename))
        # filename = secure_filename(f.filename)
        # imgX = load_image(img_path=filename)
        f.seek(0)
        img_bytes = f.read()
        if len(img_bytes) == 0:
            flash('未上传图片或图片无效，请重新上传')
            return redirect(url_for('index'))
        try:
            imgX, img_raw255 = load_image_from_bytes(img_bytes)
        except Exception as e:
            flash(f'读取文件失败，请重新上传图片\n错误：{e}')
            return redirect(url_for('index'))
        prob, pred = predict(MODEL, imgX, GRAPH)
        percentage = round(prob*100, 3)
        img_raw = b64encode(img_bytes).decode("utf-8")
        pred_str = '绿化充足' if pred==1 else '绿化不足'
        heatmap_mat = grad_cam(MODEL, imgX, GRAPH)
        # do not use matplotlib_fig as it has white space even for tight layouT
        # heatmap_file = generate_matplotlib_fig(heatmap_mat)
        # img_heatmap = b64encode(heatmap_file).decode("utf-8")
        buf_pure, buf_superimposed = show_heatmap(heatmap_mat, img_raw255)
        img_heatmap = b64encode(buf_pure).decode("utf-8")
        img_overlay = b64encode(buf_superimposed).decode("utf-8")
        return render_template('predict.html', 
                               img_raw=img_raw, 
                               img_heatmap=img_heatmap, 
                               img_overlay=img_overlay,
                               prob=percentage, 
                               pred_str=pred_str)
    else:
        flash('请首先上传图片')
        return redirect(url_for('index'))
        
@app.route("/intro", methods=['GET'])
def show_intro():
    return render_template('intro.html')
    
if __name__ == '__main__':
    global MODEL, GRAPH
    model_path = '../models/saved_model/cnn.model'
    MODEL = load_model(model_path)
    GRAPH = tf.get_default_graph() 
    app.secret_key = 'canwang'
    app.run()