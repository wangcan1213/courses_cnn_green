# -*- coding: utf-8 -*-

# 本程序需要导入以下包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob               #用于设定通配符
import h5py               #用于保存瓶颈特征
from sklearn.model_selection import train_test_split        #从sklearn中导入训练集/验证集分割模块
from sklearn.preprocessing import LabelBinarizer            #从sklearn中导入二分类编码器模块 
from sklearn.metrics import confusion_matrix                #从sklearn中导入混淆矩阵模块
from sklearn.externals import joblib                        #从sklearn中导入模型保存与读取模块

# 图像处理的相关包
from keras.preprocessing.image import load_img, img_to_array    #用于图像读取，转化为数组

# VGG16模型与Logistic回归模型的相关包
from keras import backend as K                                  #导入Keras后端
from keras.applications.vgg16 import VGG16, preprocess_input    #导入VGG16网络，以及数据预处理模块
from sklearn.linear_model import LogisticRegression             #从sklearn中导入Logistic回归模块
from keras.models import Model                                  #导入Model模块，可以通过指定输入层和输出层，设定一个模型

# 功能：从文件夹中读取图像数据，利用迁移模型得到瓶颈特征，然后按训练和测试分割
# 输入：datasetPath（格式，字符串）：图像数据存放的文件夹，下面包含若干个子文件夹，每个子文件夹内存放某种类别的图像，子文件夹的名称即为类别名
# 输入：transferModel（格式：Keras Model object）：“迁移”至本任务，用于提取瓶颈特征的迁移模型
# 输入：target_size（格式：tuple）：迁移模型的输入图像尺寸
# 输出：X_train, X_test, Y_train, Y_test（格式：numpy array）是训练集与验证集的X与Y，其中，X的维度为(nsamples, ndims)，ndims为瓶颈特征压平后的维度，Y的维度为(nsamples,)
# 输出：lb（格式：object）：用于对标签进行编码的二分类编码器（LabelBinarizer）
def extract_bottleneck_features(datasetPath, transferModel, target_size):
    labels = os.listdir(datasetPath)        #获得文件夹下面的子文件夹列表，由于子文件夹的名称即为类别名，因此该列表为类别列表
    lb = LabelBinarizer()                   #构建一个二分类编码器
    lb.fit(labels)                          #用二分类编码器对类别名称进行编码
    X = []
    Y = []
    for i, label in enumerate(labels):                                      #依次处理每个子文件夹，即每种类别
        crt_path = os.path.join(datasetPath, label)                         #获取当前子文件夹的地址
        for img_path in glob.glob(os.path.join(crt_path, '*.jpg')):         #依次处理当前子文件夹中的每张图像，通过glob设定通配符
            img = load_img(img_path, target_size=target_size)               #读取图像
            x = img_to_array(img)                                           #将图像转换为数组，此时为3维：(height,width,nchanels)
            x = np.expand_dims(x, axis=0)          #补上样本量维度，将3维数组扩充为4维：(nsamples,height,width,nchanels)，nsamples=1，即只有1个样本
            x = preprocess_input(x)                #数据预处理，将特征取值放缩至-1~1
            x = transferModel.predict(x)           #用transferModel进行预测，得到瓶颈层特征
            x = x.flatten()                        #将瓶颈层特征压平
            X.append(x)                            
            Y.append(label)
    X = np.array(X)
    Y = np.array(Y)
    Y = lb.transform(Y).ravel()             #用二分类编码器将Y中的标签转换为编码
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=1)   #以0.3的验证集比例拆分训练集与验证集，固定随机种子，使拆分结果可重现
    for (i, label) in enumerate(lb.classes_):
        print("{}. {}".format(i + 1, label))            #显示有哪些类别
    return X_train, X_test, Y_train, Y_test, lb
 
 
# 主函数
if __name__ == '__main__':
    
    #设定数据集文件夹的位置，请根据实际情况修改
    datasetPath = r'D:\02-textbook-en\cnn\dataset_600'    
    
    # 设定保存瓶颈特征的文件的路径，请根据实际情况调整
    featureFilePath = r'D:\02-textbook-en\cnn\vgg16_bottleneck_features'
    
    # 如果该路径的文件不存在，则需要提取瓶颈特征，并保存至该路径
    if not os.path.isfile(featureFilePath):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'        #为了减少多余信息的干扰，屏蔽Tensorflow通知信息的显示，只显示警告和报错信息，='0'则显示全部信息
        base_model = VGG16(weights='imagenet')          #基础模型为VGG16，加载的预训练权重是在imagenet数据集上训练的权重
        base_model.summary()                            #显示VGG16的网络结构
        
        #下面通过Keras的Model模块，创建一个以VGG16的瓶颈层为输出层的瓶颈特征提取模型
        bottleneckExtractModel = Model(input=base_model.input,             #该模型的输入为基础模型(VGG16)的输入
            output=base_model.get_layer('block5_pool').output)  #该模型的输出为基础模型(VGG16)的瓶颈层(block5_pool)的输出
        target_size = (224, 224)                    #VGG16要读取的图像尺寸为(244,244)
        
        # 利用前面定义的extract_bottleneck_features函数读取图像，提取瓶颈特征，并拆分训练/验证集
        X_train, X_test, Y_train, Y_test, lb = extract_bottleneck_features(datasetPath, bottleneckExtractModel, target_size)  

        # 保存瓶颈特征
        file = h5py.File(featureFilePath, 'w')          #用h5py包创建一个新文件
        file.create_dataset('X_train', data=X_train)    #在该文件中保存X_train，下面依次保存X_test, Y_train, Y_test
        file.create_dataset('X_test', data=X_test)
        file.create_dataset('Y_train', data=Y_train)
        file.create_dataset('Y_test', data=Y_test)
        file.close()                                   #所有的瓶颈特征和对应标签已保存，将文件关闭

    # 经过上面的if判断，保存瓶颈特征的文件已经存在，从该文件中读取瓶颈特征
    file=h5py.File(featureFilePath, 'r')
    X_train = file['X_train'][:]
    Y_train = file['Y_train'][:]
    X_test = file['X_test'][:]
    Y_test = file['Y_test'][:]
    file.close()
    
    # 输出各数据的维度，以供校核
    print('X_train shape: ' + str(X_train.shape))
    print('X_test shape: ' + str(X_test.shape))
    print('Y_train shape: ' + str(Y_train.shape))
    print('Y_test shape: ' + str(Y_test.shape))
    
    # --------以下利用瓶颈特征，对数据进行最后的训练-------- #
    
    L2 = 2**16                  # 设置L2正则化系数，需要不断尝试不同取值，以确认最佳超参数
    lrModel = LogisticRegression(C=1/L2,solver='lbfgs',max_iter=1000)       #建立一个logistic回归模型，C为正则化系数的倒数，采用lbfgs方法进行训练
    lrModel.fit(X_train, Y_train)                   #用训练集数据对Logistic回归模型进行训练
    
    train_score = lrModel.score(X_train,Y_train)    #得到训练集准确率
    test_score = lrModel.score(X_test, Y_test)      #得到验证集准确率
    print('\nTrainning acc {}'.format(train_score))
    print('Test acc {}\n'.format(test_score))

    Y_train_predict = lrModel.predict(X_train)          #得到训练集的预测结果
    Y_test_predict = lrModel.predict(X_test)            #得到验证集的预测结果
    train_matrix = confusion_matrix(Y_train,Y_train_predict)            #生成训练集混淆矩阵
    test_matrix = confusion_matrix(Y_test, Y_test_predict)              #生成验证集混淆矩阵
    print('\nTrain Confusion Matrix:')
    print(train_matrix)
    print('\nTest Confusion Matrix:')
    print(test_matrix)
    
    # 保存训练好的Logistic回归模型，以供后续使用
    modelFile = r'D:\02-textbook-en\cnn\transfer_vgg16_logisitic.model'     #设定模型保存路径，请根据实际情况修改
    joblib.dump(lrModel, modelFile)         #保存模型
    print('\n\n与VGG16对应的Logistic回归模型已被保存至以下路径：'+modelFile)    