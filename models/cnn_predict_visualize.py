# -*- coding: utf-8 -*-

# 本程序需要导入以下包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 图像处理的相关包
from keras.preprocessing.image import load_img, img_to_array            #用于图像读取，转化为数组
import cv2                                                              #opencv包，同样可以读取图像，用于可视化

# Keras模型的相关包
from keras import backend as K                   #导入Keras后端
from keras import models                         #导入models模块

if __name__ == '__main__':

    # 以下为一些文件路径的设定，请根据实际情况修改
    img_path = 'test/2.jpg'                #图像的读取路径
    img_save_path = 'test/2_cam.jpg'       #图像可视化结果的保存路径
    model_path = 'cnn/cnn.model'                  #CNN模型的读取路径
    
    # 读取图像并进行预处理，这里使用Keras自带的load_img读取，三个通道的顺序为RGB
    img = load_img(img_path, target_size=(100,200))     #读取图像，target_size用于设定图像的平面尺寸(height, width)
    X = img_to_array(img)                               #将图像转化为numpy数组
    X /= 255                                            #将像素值由0~255转换为0~1
    # 现前图像为3维数组(height, width, nchanels)，还需要补足一个维度，即样本量nsample=1
    X = np.expand_dims(X, axis=0)                       #维度扩展，使数组格式变为(nsample, height, width, nchanels)，本例中应当为(1,100,200,3)
    
    # 读取CNN模型
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'            #为了减少多余信息的干扰，屏蔽Tensorflow通知信息的显示，只显示警告和报错信息，='0'则显示全部信息
    cnnModel = models.load_model(model_path)
    
    # 进行预测
    prob = cnnModel.predict(X)                          #得到原始预测值，即sigmoid激活后输出的green=1的概率
    pred = cnnModel.predict_classes(X)                  #得到预测的类别，如果prob>0.5则预测为green=1，反之则预测为green=0
    prob, pred = prob[0][0], pred[0][0]                 #上面的prob和pred是m*n的二维数组，m为样本量，n为输出节点数量，这里只有1个样本，1个输出节点，我们可以把数组中的具体内容提出来
    if pred == 1:
        pred_str = '绿化充足'
    elif pred == 0:
        pred_str = '绿化不足'
    print('\n\n预测结果：\ngreen = {}\n类别：{}\nP(green=1)={}'.format(pred, pred_str, prob))       #打印结果

    # ---------------以下：执行Grad-CAM可视化--------------- #
    # 该方法求取输出结果对最后一个卷积层的每个chanel的梯度，作为该chanel的权重，然后对该卷积层的所有chanels进行加权平均，
    # 从而帮助我们判断为CNN是在图像的哪些位置发现了特征，是如何实现识别功能的。
    last_conv_layer = cnnModel.get_layer('conv2d_2')                                    # 获得最后一个卷积层
    grads = K.gradients(cnnModel.output, last_conv_layer.output)[0]                     # 计算梯度，此时的尺寸是四维的：(nsample,height,width,nchanels)
    pooled_grads = K.mean(grads, axis=(0,1,2))                                          # 计算每个chanel在的平均梯度，计算后的尺寸是一维的：nchanels
    # 组建一个临时模型（只是一个输入→输出的关系，暂时并没有数据），该模型输入即为cnnModel的输入（cnnModel.input），
    # 输出为两部分：其一是最后一个卷积层各chanel的权重（pooled_grads），其二是最后一个卷积层各chanel的输出（last_conv_layer.output[0]）
    iterate = K.function([cnnModel.input], [pooled_grads, last_conv_layer.output[0]])   
    pooled_grads_value, conv_layer_output_value = iterate([X])                          # 利用以上模型，对于我们的输入数据X，计算其对应的两个输出
    
    nchanels = conv_layer_output_value.shape[2]                             #获得最后一个卷积层的chanel数量
    for i in range(nchanels):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]           #将最后一个卷积层的每一个chanel乘以它对应的权重
    heatmap = np.mean(conv_layer_output_value, axis=-1)                     #现在所有chanel已经被加权，求平均值即得到heatmap数据
    heatmap = np.maximum(heatmap, 0)                                        #为了正确显示，要保证heatmap为正，因此取其与0的最大值
    heatmap /= np.max(heatmap)                                              #将heatmap变换为0-1之间
    plt.matshow(heatmap)                                                    #显示heatmap
    plt.axis('off')                                                         #不显示坐标轴
    plt.show()

    # 最后，通过opencv（cv2）将heatmap与原始图像重叠，并保存为jpg文件
    img_raw = cv2.imread(img_path)                                          #通过cv2重新读取原始图像，并保持其尺寸，注意，cv2读取的通道顺序不是RGB，而是BGR
    heatmap = cv2.resize(heatmap, (img_raw.shape[1], img_raw.shape[0]))     #将heatmap的尺寸变换为原始图像的尺寸
    heatmap = np.uint8(255*heatmap)                                         #为了正确显示，需要将heatmap由0-1变为0-255的整数，此时的heatmap是一维的，相当于灰度图
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)                  #将heatmap变换为彩色图，采用JET配色方案
    superimposed_img = heatmap*0.4 + img_raw                                #将heatmap与原始图像进行叠合
    
    cv2.imwrite(img_save_path, superimposed_img)                            #将叠合图像保存至jpg文件
    print('\n\nCAM可视化结果已保存。')
    # ---------------以上：执行Grad-CAM可视化--------------- #





