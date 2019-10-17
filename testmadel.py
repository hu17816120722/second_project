import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt

from keras.utils.vis_utils import plot_model
import pydot_ng as pydot

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
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
# 每个输入通道的大小是519位，一共9个通道
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
y_train = np_utils.to_categorical(y_train, nb_class)
y_valid = np_utils.to_categorical(y_valid, nb_class)

# 载入模型
model = load_model('model.h5')

# 评估模型
loss,accuracy = model.evaluate(X_valid_r,y_valid)

print('\ntest loss',loss)
print('accuracy',accuracy)
# 训练模型
model.fit(X_train_r, y_train, nb_epoch=2, validation_data=(X_valid_r, y_valid), batch_size=16)

# 保存参数，载入参数
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')
# 保存网络结构，载入网络结构
from keras.models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)

print(json_string)