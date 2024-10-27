from  keras.datasets import mnist
import keras #深度学习框架
import keras.models  #模型
from  keras.models import  Sequential #神经网络
from  keras.layers  import Dense  ,Dropout,Flatten  #处理神经网络层
from keras.layers import Conv2D,MaxPooling2D #处理平面数据
from keras  import  backend as K  #处理结束

batch_size=128 #批量操作大小
num_classes=10 #识别的结果
epochs=12 #训练次数
#图片大小
img_rows,img_cols=28,28
(x_train,y_train),(x_test,y_test)=mnist.load_data() #载入数据
if K.image_data_format()=="channels_first": #格式-数据格式
    x_train=x_train.reshape(x_train.shape[0],1,img_rows,img_cols) #形状调整
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape=(1,img_rows,img_cols) #y用于训练数据形状1*28*28
else:
    x_train = x_train.reshape(x_train.shape[0],  img_rows, img_cols,1)  # 形状调整
    x_test = x_test.reshape(x_test.shape[0],  img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols,1)  # y用于训练数据形状1*28*28
x_train=x_train.astype("float32")
x_test=x_test.astype("float32")
x_train /=255 #xtrain矩阵的浮点类型，0-255  256   ，颜色值改成203,0.78
x_test /=255
print(x_train.shape[0],"样本数量")
print(x_test.shape[0],"样本数量")
print(x_train.shape,x_test.shape)

#结果数据，0-9，10类，分类10个输出结果
y_train=keras.utils.np_utils.to_categorical(y_train,num_classes)
y_test=keras.utils.np_utils.to_categorical(y_test,num_classes)

model=Sequential() #新建一个神经网络
model.add(Conv2D(32,
                 activation="relu",
                 input_shape= input_shape,
                 nb_row=3,
                 nb_col=3
                 ))
model.add(Conv2D(64,
                 activation="relu",
                 nb_row=3,
                 nb_col=3
                 ))
model.add(MaxPooling2D(pool_size=(2,2))) #搜索最有结果
model.add(Dropout(0.35)) #输出系数
model.add(Flatten()) #平整处理数据

model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5)) #输出系数

model.add(Dense(num_classes,activation="softmax")) #结果

model.compile(loss=keras.metrics.categorical_crossentropy, #训练的损失函数
              optimizer=keras.optimizers.Adadelta(), #优化
              metrics=["accuracy"]) #精确
model.fit(x_train,y_train,
          batch_size=batch_size,#批量处理的数量
          epochs=epochs, #训练次数
          verbose=1, #行为区别
          validation_data=(x_test,y_test)) #验证的数据
score=model.evaluate(x_test,y_test,verbose=0) #评分
print(score) #识别的分数