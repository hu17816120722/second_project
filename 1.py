import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# StratifiedShuffleSplit可以用来把数据集洗牌，并拆分成训练集和验证集
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt

from keras.utils import to_categorical

from keras.utils.vis_utils import plot_model
import pydot_ng as pydot

from sklearn.metrics import confusion_matrix
import seaborn as sns
pydot.Dot.create(pydot.Dot())
# 每个通道173个比特位
train = pd.read_csv('./train1.csv')
test = pd.read_csv('./test1.csv')
def encode(train, test):
    # 用LabelEncoder种类标签编码，labels对象是训练集上的标签列表
    label_encoder = LabelEncoder().fit(train.species)
    labels = label_encoder.transform(train.species)
    classes = list(label_encoder.classes_)
    # 此处把不必要的训练集和测试集的列删除
    train = train.drop('species', axis=1)
    test = test.drop('species', axis=1)
    return train, labels, test, classes
train, labels, test, classes = encode(train, test)
# 这里只是标准化训练集的特征值
scaler = StandardScaler().fit(train.values)
scaled_train = scaler.transform(train.values)
# 把数据集拆分成训练集和测试集，测试集占10%
sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
for train_index, valid_index in sss.split(scaled_train, labels):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = labels[train_index], labels[valid_index]
# 每个输入通道的大小是173位，一共9个通道
nb_features = 173
nb_class = len(classes)
#  把输入数据集reshape成keras喜欢的格式：（样本数，通道大小，通道数）
X_train_r = np.zeros((len(X_train), nb_features, 9))
# 这里的做法是先把所有元素初始化成0之后，再把刚才的数据集中的数据赋值过来
X_train_r[:, :, 0] = X_train[:, :nb_features]
X_train_r[:, :, 1] = X_train[:, nb_features:346]
X_train_r[:, :, 2] = X_train[:, 346:519]
X_train_r[:, :, 3] = X_train[:, 519:692]
X_train_r[:, :, 4] = X_train[:, 692:865]
X_train_r[:, :, 5] = X_train[:, 865:1038]
X_train_r[:, :, 6] = X_train[:, 1038:1211]
X_train_r[:, :, 7] = X_train[:, 1211:1384]
X_train_r[:, :, 8] = X_train[:, 1384:]
# 验证集也要reshape一下
X_valid_r = np.zeros((len(X_valid), nb_features, 9))
X_valid_r[:, :, 0] = X_valid[:, :nb_features]
X_valid_r[:, :, 1] = X_valid[:, nb_features:346]
X_valid_r[:, :, 2] = X_valid[:, 346:519]
X_valid_r[:, :, 3] = X_valid[:, 519:692]
X_valid_r[:, :, 4] = X_valid[:, 692:865]
X_valid_r[:, :, 5] = X_valid[:, 865:1038]
X_valid_r[:, :, 6] = X_valid[:, 1038:1211]
X_valid_r[:, :, 7] = X_valid[:, 1211:1384]
X_valid_r[:, :, 8] = X_valid[:, 1384:]
# 下面是Keras的一维卷积实现，原作者尝试过多加一些卷积层，
# 结果并不能提高准确率，可能是因为其单个通道的信息本来就太少，深度太深的网络本来就不适合
model = Sequential()
# 一维卷积层用了512个卷积核，输入是64*3的格式
# 此处要注意，一维卷积指的是卷积核是1维的，而不是卷积的输入是1维的，1维指的是卷积方式
model.add(Convolution1D(nb_filter=512, filter_length=1, input_shape=(nb_features, 9)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.2))##原来0.4
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(nb_class))
# softmax经常用来做多类分类问题
model.add(Activation('softmax'))

y_train = np_utils.to_categorical(y_train, nb_class)
y_valid = np_utils.to_categorical(y_valid, nb_class)
sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.summary()
nb_epoch = 2
history=model.fit(X_train_r, y_train, nb_epoch=nb_epoch, validation_data=(X_valid_r, y_valid), batch_size=16)

model.save('model.h5')
plot_model(model, to_file='model.png', show_shapes=True)
score = model.evaluate(X_train_r,y_train, verbose=0)
history_dict=history.history
loss_value=history_dict["loss"]
val_loss_value=history_dict["val_loss"]
epochs=range(1,len(loss_value)+1)
plt.plot(epochs,loss_value,"bo",label="Training loss")
plt.plot(epochs,val_loss_value,"b",label="Validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()


from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import numpy as np
import tensorflow as tf
y_pre=model.predict_classes(X_valid_r)
a = [str(i) for i in y_pre]
y_valid1 = np.array([np.argmax(one_hot)for one_hot in y_valid])
b = [str(i) for i in y_valid1]
labels_name = ['0','1','2','3','4']
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.YlGn)    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm = confusion_matrix(b, a)
print(cm)
plot_confusion_matrix(cm, labels_name, "Confusion Matrix")
plt.savefig('cm.png', format='png')
plt.show()