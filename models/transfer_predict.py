# -*- coding: utf-8 -*-

# 本程序需要导入以下包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib            #从sklearn中导入模型保存与读取模块

# 图像处理的相关包
from keras.preprocessing.image import load_img, img_to_array        #用于图像读取，转化为数组

# Keras模型的相关包
from keras import backend as K                                      #导入Keras后端
from keras.applications.vgg16 import VGG16, preprocess_input        #导入VGG16网络，以及数据预处理模块
from keras.models import Model                                     #导入Model模块，可以通过指定输入层和输出层，设定一个模型

# 主函数
if __name__ == '__main__':

    # 以下为一些文件路径的设定，请根据实际情况修改
    img_path = r'D:\02-textbook-en\cnn\test\2.jpg'                #图像的读取路径
    lr_model_path = r'D:\deep-learning\jiejing\good_img\transfer_vgg16_logisitic.model'             #VGG16对应的Logistic回归模型的读取路径
    
    
    # 读取图像并进行预处理，这里使用Keras自带的load_img读取，三个通道的顺序为RGB
    img = load_img(img_path, target_size=(224,224))     #读取图像，target_size用于设定图像的平面尺寸(height, width)
    X = img_to_array(img)                               #将图像转化为numpy数组
    # 现前图像为3维数组(height, width, nchanels)，还需要补足一个维度，即样本量nsample=1
    X = np.expand_dims(X, axis=0)                       #维度扩展，使数组格式变为(nsample, height, width, nchanels)，本例中应当为(1,100,200,3)
    X = preprocess_input(X)
    
    # 创建以VGG16的瓶颈层为输出层的瓶颈特征提取模型
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'        #为了减少多余信息的干扰，屏蔽Tensorflow通知信息的显示，只显示警告和报错信息，='0'则显示全部信息
    base_model = VGG16(weights='imagenet')              #基础模型为VGG16，加载的预训练权重是在imagenet数据集上训练的权重
    bottleneckExtractModel = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output)
    bottleneckFeatures = bottleneckExtractModel.predict(X)      #用瓶颈特征提取模型对X进行预测，得到X对应的瓶颈特征
    bottleneckFeatures = bottleneckFeatures.flatten()           #将瓶颈特征压平至一维
    bottleneckFeatures = np.expand_dims(bottleneckFeatures, axis=0) #加入样本量维度，将维度扩展至二维
    
    # 读取Logistic回归模型
    lrModel = joblib.load(lr_model_path)      #读取模型
    
    # 进行预测
    prob = lrModel.predict_proba(bottleneckFeatures)            #得到原始预测值，即sigmoid激活后输出的green=1的概率
    pred = lrModel.predict(bottleneckFeatures)                  #得到预测的类别，如果prob>0.5则预测为green=1，反之则预测为green=0
    prob, pred = prob[0][1], pred[0]        #prob是m*n的数组，m为样本量，n为类别数，prob[0][1]为第1个样本（只有一个样本）、第2个类别（green=1）的概率
    if pred == 1:
        pred_str = '绿化充足'
    elif pred == 0:
        pred_str = '绿化不足'
    print('\n\n预测结果：\ngreen = {}\n类别：{}\nP(green=1)={}'.format(pred, pred_str, prob))       #打印结果




