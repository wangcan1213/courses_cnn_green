# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from imageio import imread
os.environ['KMP_WARNINGS'] = '0'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
import cv2
from keras import backend as K
from keras import models

def load_model(model_path):
    return models.load_model(model_path)
    
def load_image_from_bytes(img_bytes, target_size=(100,200)):
    img = imread(img_bytes)
    img_raw = np.uint8(img)
    imgX = resize(img, target_size)  # also convert to 0-1
    if imgX.shape[-1] > 3:
        imgX = imgX[:,:,:3]
    imgX = np.expand_dims(imgX, axis=0)
    return imgX, img_raw
    
def load_image_from_path(img_path, target_size=(100,200)):
    # in this way we cannot know the raw shape
    img = load_img(img_path, target_size=target_size)
    imgX = img_to_array(img)
    imgX /= 255
    if imgX.shape[-1] > 3:
        imgX = imgX[:,:,:3]
    imgX = np.expand_dims(imgX, axis=0)
    return imgX
  
  
def predict(cnnModel, imgX, graph):
    with graph.as_default():
        prob = cnnModel.predict(imgX)
        pred = cnnModel.predict_classes(imgX)
    prob, pred = prob[0][0], pred[0][0]
    return prob, pred 


def grad_cam(cnnModel, imgX, graph):
    with graph.as_default():
        last_conv_layer = cnnModel.get_layer('conv2d_2')
        grads = K.gradients(cnnModel.output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0,1,2))
        
        iterate = K.function([cnnModel.input], [pooled_grads, last_conv_layer.output[0]])   
        pooled_grads_value, conv_layer_output_value = iterate([imgX])
        nchanels = conv_layer_output_value.shape[2]
        for i in range(nchanels):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap_mat = np.mean(conv_layer_output_value, axis=-1)
    heatmap_mat = np.maximum(heatmap_mat, 0)
    heatmap_mat /= np.max(heatmap_mat)
    return heatmap_mat

    
def show_heatmap(heatmap_mat, img_raw):
    # using cv2
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
    heatmap_mat = cv2.resize(heatmap_mat, (img_raw.shape[1], img_raw.shape[0]))
    heatmap_mat = np.uint8(255 * heatmap_mat)
    heatmap_mat = cv2.applyColorMap(heatmap_mat, cv2.COLORMAP_JET)
    superimposed_img = heatmap_mat*0.4 + img_raw
    # img_str = cv2.imencode('.jpg', superimposed_img)[1].tostring()
    _, buf_pure = cv2.imencode('.jpg', heatmap_mat)
    _, buf_superimposed = cv2.imencode('.jpg', superimposed_img)
    return buf_pure, buf_superimposed

    
def generate_matplotlib_fig(heatmap_mat):
    plt.figure()
    plt.matshow(heatmap_mat)
    plt.axis('off')   
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.read()