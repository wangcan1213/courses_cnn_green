# -*- coding: utf-8 -*-

# 本程序需要导入以下包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob                                                 #glob包用于设定通配符
from sklearn.model_selection import train_test_split        #从sklearn中导入训练集/验证集分割模块
from sklearn.preprocessing import LabelBinarizer            #从sklearn中导入二分类编码器模块
from sklearn.metrics import confusion_matrix                #从sklearn中导入混淆矩阵模块

# 图像处理的相关包
from keras.preprocessing.image import load_img, img_to_array    #用于图像读取，转化为数组
from keras.preprocessing.image import ImageDataGenerator        #用于图像数据增强

# Keras包中与CNN相关的模块
from keras import backend as K                                              #导入Keras后端
from keras.models import Sequential                                         #导入Sequential模块，用于组建顺序网络结构
from keras.layers import Conv2D, MaxPooling2D                               #导入卷积层和最大池化层模块
from keras.layers import Dense, Dropout, Flatten, Input, Activation         #导入全连接层、dropout层、压平层、输入层、激活层模块
from keras.optimizers import Adam                                           #导入Adam训练器
from keras.regularizers import l2                                           #导入L2正则化

# 正确显示中文字体
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 设置超参数，可以根据需求进行调整
EPOCHS = 40                         #训练的代数（epochs）
INIT_LR = 1e-4                      #初始学习率
BS = 32                             #mini-batch size
IMAGE_DIMS = (100, 200, 3)          #图像尺寸(height,width,nchanels)
REG = 0.000                         #L2正则化系数，这里设置为0，相当于未使用

# 处理随机性：虽然下面设定了numpy和tensorflow中的随机种子，使训练结果尽可能相似，但是GPU的使用还会带来额外的随机性，多次运行的结果并非完全相同
np.random.seed(1)                           #设定numpy中的随机种子
from tensorflow import set_random_seed      #引入tensorflow中的随机种子
set_random_seed(2)                          #设定tensorflow中的随机种子

# 功能：从文件夹中读取图像数据，按训练和测试分割
# 输入：datasetPath（格式，字符串）：图像数据存放的文件夹，下面包含若干个子文件夹，每个子文件夹内存放某种类别的图像，子文件夹的名称即为类别名
# 输出：X_train, X_test, Y_train, Y_test（格式：numpy array）是训练集与验证集的X与Y，其中，X的维度为(nsamples, height, width, nchanels)，Y的维度为(nsamples,)
# 输出：lb（格式：object）：用于对标签进行编码的二分类编码器（LabelBinarizer）
def read_data(datasetPath):
    labels = os.listdir(datasetPath)                    #获得文件夹下面的子文件夹列表，由于子文件夹的名称即为类别名，因此该列表为类别列表
    lb = LabelBinarizer()                               #构建一个二分类编码器
    lb.fit(labels)                                      #用二分类编码器对类别名称进行编码
    X = []
    Y = []
    for i, label in enumerate(labels):                                              #依次处理每个子文件夹，即每种类别
        crt_path = os.path.join(datasetPath, label)                                 #获取当前子文件夹的地址
        for img_path in glob.glob(os.path.join(crt_path, '*.jpg')):                 #依次处理当前子文件夹中的每张图像，通过glob设定通配符
            img = load_img(img_path)                                                #读取图像
            x = img_to_array(img)                                                   #将图像转换为数组
            X.append(x)                                                 
            Y.append(label)
    X = np.array(X) / 255                                                           #将X由0-255转换为0-1
    Y = np.array(Y)                     
    Y = lb.transform(Y)                                                           #用二分类编码器将Y中的标签转换为编码
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=1)     #以0.3的验证集比例拆分训练集与验证集，固定随机种子，使拆分结果可重现
    for (i, label) in enumerate(lb.classes_):
        print("{}. {}".format(i + 1, label))                                        #显示有哪些类别
    return X_train, X_test, Y_train, Y_test, lb
        
    
# 功能：根据自定义的结构，建立一个卷积神经网络
# 输入：width, height, depth（格式：整数）：输入图片的宽度、高度、通道数（深度）
# 输出：model（格式：Keras object）是一个待训练的卷积神经网络 
def CNNNet(width, height, depth):
    model = Sequential()                                    #模型为Sequential顺序模型，即按照顺序不断地加入各个层
    
    model.add(Conv2D(32, (3, 3),                            #加入第一个卷积层，包含32个(3,3)的卷积核
        padding='same',                                     #padding='same'表示通过padding使输出尺寸与原尺寸一致，默认strides=1
        input_shape=(height, width, depth),                 #由于是第一层，因此还需要指定输入图像的尺寸
        kernel_regularizer=l2(REG)))                        #以REG为正则化系数，对参数进行L2正则化
        
    model.add(Activation('relu'))                           #增加relu激活层，对上一层的结果进行激活
    model.add(MaxPooling2D(pool_size=(2, 2)))               #增加第一个max pooling层，窗口尺寸为(2,2)，strides默认与窗口尺寸相同             
    
    model.add(Conv2D(64, (3, 3),padding='same',             #增加第二个卷积层，包含64个(3,3)卷积核，并使输出尺寸与原尺寸一致
        kernel_regularizer=l2(REG)))                        #以REG为正则化系数，对参数进行L2正则化 
        
    model.add(Activation('relu'))                           #对上一层结果进行relu激活
    model.add(MaxPooling2D(pool_size=(2, 2)))               #增加第二个max pooling层
    
    model.add(Flatten())                                    #将3维图像压平成1维
    
    model.add(Dense(64,                                     #加入全连接层，包含64个节点
        kernel_regularizer=l2(REG)))                        #以REG为正则化系数，对参数进行L2正则化
        
    model.add(Activation('relu'))                           #对上一层结果进行relu激活
    model.add(Dropout(0.5))                                 #对上一层结果进行dropout正则化，p=0.5
    model.add(Dense(1))                                     #由于是二分类，输出层可以只有一个节点
    model.add(Activation('sigmoid'))                        #对输出层进行sigmoid激活
    return model
    

# 功能：对卷积神经网络的训练历史过程（训练/验证准确率）绘制折线图
# 输入：history（格式：pandas DataFrame）：包含训练/验证准确率的数据表
# 输入：ma（格式：0-1的浮点数）：平滑系数，越大则平滑程度越大，=0时不进行平滑，直接使用原始数据  
def history_line_plot(history, sm=0.6):
    acc_raw = history['acc']                                #训练准确率的原始值
    val_acc_raw = history['val_acc']                        #验证准确率的原始值
    acc_sm = []                                             #用于存储训练准确率的平滑值 
    val_acc_sm = []                                         #用于存储验证准确率的平滑值
    n = len(acc_raw)                                        #迭代次数（epochs）
    iter = [i+1 for i in range(n)]                          #生成从1到n的数字序列：1,2,3,4,...,n
    
    for i in range(n):
        if i == 0:
            #在第一次迭代时，平滑值等于原始值
            acc_sm.append(acc_raw[0])                       
            val_acc_sm.append(val_acc_raw[0])
        else:
            #在下面的迭代中，当次迭代的平滑值 = sm*当次迭代的原始值 + (1-sm)*上一次迭代的平滑值，sm为平滑系数
            acc_sm.append(acc_raw[i]*sm + (1-sm)*acc_sm[i-1])
            val_acc_sm.append(val_acc_raw[i]*sm + (1-sm)*val_acc_sm[i-1])
    
    # 在1行2列的图阵中，以第1个图（左图）对训练误差绘图
    ax1 = plt.subplot(121)
    ax1.plot(iter, acc_raw, color=[1,.4,.2], lw=2, alpha=0.2, label='原始值')
    ax1.plot(iter, acc_sm, color=[1,.4,.2], lw=2, label='平滑值')
    plt.xlabel('Epochs')
    plt.ylabel('准确率')
    plt.title('训练集')
    plt.grid()
    plt.legend(loc='lower right')
    [ymin1, ymax1] = ax1.get_ylim()             #获得训练误差图y轴的取值范围
    
    # 在1行2列的图阵中，以第2个图（右图）对训练误差绘图
    ax2 = plt.subplot(122)
    ax2.plot(iter, val_acc_raw, color=[0,.45,.75], lw=2, alpha=0.2, label='原始值')
    ax2.plot(iter, val_acc_sm, color=[0,.45,.75], lw=2, label='平滑值')
    plt.xlabel('Epochs')
    plt.ylabel('准确率')
    plt.title('验证集')
    plt.grid()
    plt.legend(loc='lower right')
    [ymin2, ymax2] = ax2.get_ylim()             #获得验证误差图y轴的取值范围
    ymin = min(ymin1, ymin2)                    #取两图中y轴的最小值为公用最小值
    ymax = max(ymax1, ymax2)                    #取两图中y轴的最大值为公用最大值 
    ax1.set_ylim([ymin, ymax])                  #重置训练误差图中y轴的取值范围为[公用最小值，公用最大值]，以便于比较
    ax2.set_ylim([ymin, ymax])                  #重置验证误差图中y轴的取值范围为[公用最小值，公用最大值]，以便于比较
    
    plt.show()
    
        
# 主函数
if __name__ == '__main__':
    
    #设定数据集文件夹的位置，请根据实际情况修改
    datasetPath = 'dataset_600' 
    
    X_train, X_test, Y_train, Y_test, lb = read_data(datasetPath)       #从数据集文件夹中读取数据，并进行训练集/验证集的拆分
    #输出训练/验证集X/Y的尺寸
    print('X_train shape: ' + str(X_train.shape))
    print('X_test shape: ' + str(X_test.shape))
    print('Y_train shape: ' + str(Y_train.shape))
    print('Y_test shape: ' + str(Y_test.shape))

    #用ImageDataGenerator定义一个图像生成器，执行图像数据增强
    aug = ImageDataGenerator(rotation_range=5, width_shift_range=0.1,       #rotation_range为旋转角度范围，width_shift_range为水平平移的比例范围
        height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,            #height_shift_range为竖向平移的比例范围，shear_range为错切变换的比例范围，zoom_range为放缩比例范围
        horizontal_flip=True, fill_mode="nearest")                          #horizontal_flip=True为允许水平翻转，fill_mode用于设定变换后空白像素的填充方法，这里采用最近像素法
        
    model = CNNNet(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],depth=IMAGE_DIMS[2])       #用前面定义的CNNNet创建一个自定义结构的卷积神经网络模型，此时的模型只是一个结构，尚未被训练
    model.summary()                                                         #通过summary方法查看模型结构
    #创建用于训练的优化器，训练方法为Adam，设置初始学习率lr=INIT_LR，设置每次参数更新后的学习率衰减值为decay=INIT_LR / EPOCHS，即代数越大，学习率越小，Adam的其他参数（beta_1，beta_2等）一般不需要调整
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)          
    #上面的model只定义了结构，下一行用compile方法对其进行编译，要求设定损失函数（loss）、优化器（optimizer）、性能指标（metrics）
    model.compile(loss="binary_crossentropy",           #损失函数为二分类交叉熵，交叉熵是分类问题一般使用的损失函数
        optimizer=opt,                                  #优化器（即训练算法）使用上面定义的Adam
        metrics=["accuracy"])                           #性能指标使用准确率，这是分类问题最常用的性能指标之一

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'            #为了减少多余信息的干扰，屏蔽Tensorflow通知信息的显示，只显示警告和报错信息，='0'则显示全部信息
    print("[INFO] training network...")                 #打印信息：开始训练模型
    H = model.fit_generator(                            #使用fit_generator方法，生成数据增加后的图像并训练模型
        aug.flow(X_train, Y_train, batch_size=BS),      #使用上面定义的aug图像生成器，生成对训练集(X_train, Y_train)进行数据增强，mini-batch size为BS 
        validation_data=(X_test, Y_test),               #使用验证集(X_test, Y_test)进行验证
        epochs=EPOCHS, verbose=1, shuffle=False)        #epochs设定总代数，verbose=1显示详细日志，shuffle=False设定在每轮迭代前不对数据进行随机混洗
      
      
    # 保存训练好的模型
    model_save_path = 'cnn/cnn.model'        #设定模型保存路径，请根据实际情况修改
    model.save(model_save_path)                                             #保存模型
    
    # 将数据代回模型，输出混淆矩阵
    Y_train_predict = model.predict_classes(X_train)                    #用predict_classes方法预测训练集标签
    Y_test_predict = model.predict_classes(X_test)                      #预测验证集标签
    train_matrix = confusion_matrix(Y_train,Y_train_predict)            #生成训练集混淆矩阵
    test_matrix = confusion_matrix(Y_test, Y_test_predict)              #生成验证集混淆矩阵
    print('\nTrain Confusion Matrix:')
    print(train_matrix)
    print('\nTest Confusion Matrix:')
    print(test_matrix)
    
    # 对训练历史过程绘制折线图
    train_history = pd.DataFrame(H.history)                             #用history方法得到训练历史（准确率、损失函数的变化），并转化为pandas DataFrame
    history_line_plot(train_history)                                    #用history_line_plot函数绘图