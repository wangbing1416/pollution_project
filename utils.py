import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
# import matplotlib.pyplot as plt
# import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
# import os
import joblib


# 读取数据 (10个站点)
# sites = {}
# for i in range(10):
#     site = pd.read_excel('./data/testdata/site' + str(i) + '.xlsx')
#     sites[i] = site

# 对数据缺失值处理
# for i in range(10):
#     sites[i].interpolate('linear', inplace=True)  # 补全所有缺失值
#     sites[i].drop(index=[0], inplace=True)


# for i in range(10):
#     print(len(sites[i]))
# 34668
# 34668
# 34668
# 1716
# 34668
# 34668
# 34668
# 34668
# 34668
# 34668


# 分割训练集、测试集数据
def generate_dataset(site, site_index):
    '''
    传入一个站点的数据集、切割数据集,传入站点的序号(0-9)
    返回训练集、测试集
    '''
    l = 34668  # len(sites) = 34668
    training_set = site.iloc[0:l - 365 * 24, 2:]  # iloc只取行号，而非index
    test_set = site.iloc[l - 365 * 24:, 2:]

    # 对训练集和测试集数据进行归一化
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    test_set = sc.transform(test_set)

    # 存 scaler
    joblib.dump(sc, './scaler/sc' + str(site_index) + '.pkl')

    return training_set_scaled, test_set


# 暂时
# training_set_scaled, test_set = generate_dataset(sites[0])


# 初始化训练集、测试集函数
def process_data(data_scaled, time_step=24):
    '''
    time_step: 使用 time_step 个时间步用于预测，default:24
    data_scaled: 经过归一化的数据、传入训练集或测试集
    返回值: x,y为 RNN-based model 输入数据的维度要求
    '''

    # 初始化训练集
    x = []
    y = []

    # x
    for i in range(time_step, len(data_scaled)):
        x.append(data_scaled[i - time_step:i, 0])

    # y
    for i in range(time_step + 24, len(data_scaled)):
        y.append(data_scaled[i - time_step:i, 0])

    x = x[0:len(y)]

    # 对训练集进行打乱
    np.random.seed(7)
    np.random.shuffle(x)
    np.random.seed(7)
    np.random.shuffle(y)
    tf.random.set_seed(7)

    # 将训练集由list格式变为array格式
    x, y = np.array(x), np.array(y)

    # 使 x 符合 RNN 输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
    x = np.reshape(x, (x.shape[0], time_step, 1))

    return x, y


# x_train, y_train = process_data(training_set_scaled)
# x_test, y_test = process_data(test_set)


# 创建模型
def create_model():
    model = tf.keras.Sequential([
        LSTM(80, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(24)  # 被预测的 24
    ])
    return model


# model = create_model()

# model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#               loss='mean_squared_error')

# checkpoint_save_path = "./checkpoint/predict.ckpt"

# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------load the model-----------------')
#     model.load_weights(checkpoint_save_path)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                                                  save_weights_only=True,
#                                                  save_best_only=True,
#                                                  monitor='val_loss')

# history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test), validation_freq=1,
#                     callbacks=[cp_callback])

# model.summary()

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

# 测试集输入模型进行预测
# predicted_pm25 = model.predict(x_test)

# 对预测数据还原---从（0，1）反归一化到原始范围
# sc = joblib.load('./scaler/sc')
# predicted_pm25 = sc.inverse_transform(predicted_pm25)

# 对真实数据还原---从（0，1）反归一化到原始范围
# real_pm25 = sc.inverse_transform(y_test)


def get_rmse(predicted_pm25, real_pm25):
    predicted_pm25_T = predicted_pm25.T
    real_pm25_T = real_pm25.T

    rmse = []
    for i in range(len(predicted_pm25_T)):
        rmse.append(math.sqrt(mean_squared_error(predicted_pm25_T[i], real_pm25_T[i])))

    return rmse


def get_mae(predicted_pm25, real_pm25):
    predicted_pm25_T = predicted_pm25.T
    real_pm25_T = real_pm25.T

    mae = []
    for i in range(len(predicted_pm25_T)):
        mae.append((mean_squared_error(predicted_pm25_T[i], real_pm25_T[i])))

    return mae

# rmse = get_rmse(predicted_pm25, real_pm25)
# print(rmse)

# rmse_np = np.array(rmse)
# np.save("./result/rmse.npy", rmse_np)

# result = np.load('result.npy')
