from utils import *
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据 (10个站点)
sites = {}
for i in range(1):
    site = pd.read_excel('./data/site' + str(i) + '.xlsx')
    sites[i] = site

# 对数据缺失值处理
for i in range(1):
    sites[i].interpolate('linear', inplace=True)  # 补全所有缺失值
    sites[i].drop(index=[0], inplace=True)

if input("train or test") == "train":
    # 训练
    for i in range(1):

        # site3 数据量有问题(少)，不跑
        if i == 3:
            continue

        training_set_scaled, test_set = generate_dataset(sites[i], i)

        x_train, y_train = process_data(training_set_scaled)
        x_test, y_test = process_data(test_set)

        model = create_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='mean_squared_error')

        BATCH_SIZE = 32
        EPOCHS = 60

        history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test),
                            validation_freq=1)
        model.save('./model/site' + str(i) + '.h5')
        # 读取模型的方法
        # model = tf.keras.models.load_model('./model/site' + str(i) + '.h5')
        model.summary()

        # 测试集输入模型进行预测
        predicted_pm25 = model.predict(x_test)

        # 对预测数据还原---从（0，1）反归一化到原始范围
        sc = joblib.load('./scaler/sc' + str(i) + '.pkl')

        # print(predicted_pm25.shape)

        predicted_pm25 = sc.inverse_transform(predicted_pm25)

        # 对真实数据还原---从（0，1）反归一化到原始范围
        real_pm25 = sc.inverse_transform(y_test)

        rmse = get_rmse(predicted_pm25, real_pm25)
        print(rmse)

        rmse_np = np.array(rmse)
        np.save("./result/rmse/rmse" + str(i) + ".npy", rmse_np)

        mae = get_mae(predicted_pm25, real_pm25)
        print(mae)

        mae_np = np.array(mae)
        np.save("./result/mae/mae" + str(i) + ".npy", rmse_np)
else:
    # 只测试不训练
    for i in range(1):

        # site3 数据量有问题(少)，不测试
        if i == 3:
            continue

        training_set_scaled, test_set = generate_dataset(sites[i], i)

        x_train, y_train = process_data(training_set_scaled)
        x_test, y_test = process_data(test_set)

        # model = create_model()
        # model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
        #               loss='mean_squared_error')

        # BATCH_SIZE = 32
        # EPOCHS = 20

        # history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test), validation_freq=1)
        # model.save('./model/site' + str(i) + '.h5')
        # 读取模型的方法
        model = tf.keras.models.load_model('./model/site' + str(i) + '.h5')
        # model.summary()

        # 测试集输入模型进行预测
        predicted_pm25 = model.predict(x_test)

        # 对预测数据还原---从（0，1）反归一化到原始范围
        sc = joblib.load('./scaler/sc' + str(i) + '.pkl')

        # print(predicted_pm25.shape)

        predicted_pm25 = sc.inverse_transform(predicted_pm25)

        # 对真实数据还原---从（0，1）反归一化到原始范围
        real_pm25 = sc.inverse_transform(y_test)


        plt.plot(predicted_pm25.T[0], label='predicted')
        plt.plot(real_pm25.T[0], label='real')
        plt.title('predicted vs real')
        plt.legend()
        plt.show()

        rmse = get_rmse(predicted_pm25, real_pm25)
        print(rmse)

        # rmse_np = np.array(rmse)
        # np.save("./result/rmse/rmse" + str(i) + ".npy", rmse_np)

        mae = get_mae(predicted_pm25, real_pm25)
        print(mae)
