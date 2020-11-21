# import numpy as np
#
# results = []
# for i in range(10):
#     if i == 3:
#         continue
#     result = np.load('./result/rmse'+str(i)+'.npy')
#     results.append(result)
#
# print(len(results))
# print(results)

import tensorflow as tf

# 读取模型的方法
model = tf.keras.models.load_model('./model/site' + str(0) + '.h5')
