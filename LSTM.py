# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import matplotlib.pyplot as plt
# from tensorflow.keras.callbacks import LambdaCallback
#
# # 读取CSV文件
# file_path = '/Users/sunshuo/Desktop/LSTM/data/4_final_data/user1_LSTMData.csv'
# df = pd.read_csv(file_path, header=None, names=['value'])#value为列名，原始数据
#
# # 数据预处理
# scaler = MinMaxScaler(feature_range=(0, 1))#将数据缩放到0-1范围
# df['scaled_value'] = scaler.fit_transform(df['value'].values.reshape(-1, 1))#将一维数组转化为二维数组
#
# # 添加时间列
# df['time'] = np.arange(0, len(df))#时间列
#
# print(df)#最终这个DataFrame包含三列：1.原始数据value 2.归一化后的数据scaled_value, 3.时间time
#
# # 创建训练数据
# look_back = 10
# X, y, time = [], [], []
#
# for i in range(len(df) - look_back):#遍历范围是0-（长度-滑动窗口）
#     X.append(df[['scaled_value', 'time']][i:(i + look_back)].values)#x是(滑动窗口个数据)，存储的是scaled_value和time
#     y.append(df['scaled_value'][i + look_back])#y只存储下一时刻的目标变量，只是一个值
#     time.append(df['time'][i + look_back])#time只存储下一时刻的时间time，也只是一个值
#
# X, y, time = np.array(X), np.array(y), np.array(time)#将数据转换为numpy数组，有利于后边计算
#
# last_value = time[-1]
# print("最后一个值:", last_value)
#
# # 将数据转换成LSTM模型的输入形状 (samples, time steps, features)
# X = np.reshape(X, (X.shape[0], look_back, 2))  # 2表示两个特征，scaled_value和time
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(X, y, time, test_size=0.2, random_state=42)
#
# # 构建LSTM模型
# model = Sequential()
# model.add(LSTM(units=50, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dense(units=1))
# model.compile(optimizer='adam', loss='mean_squared_error')
#
#
# # 训练模型
# model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)#这个model.fit函数就可以执行30次的大epoch循环，而不用写for循环等了
#
# # 在测试集上评估模型
# loss = model.evaluate(X_test, y_test, verbose=2)
# print(f'Mean Squared Error on Test Set: {loss}')#这个loss可以打印一个模版中的内容，是model.evaluate配置好的
#
# #一下这个循环可以写在PPO中，用于生成用户未来数据
#
# # 主循环
# num_steps = 3000  # 设定主循环的总步数
# current_time = time_test[0] + 1  # 从第31ms开始
# output_file_path = 'predicted_values.csv'  # 输出文件路径
#
# with open(output_file_path, 'w') as output_file:
#     output_file.write('Time,Predicted_Value\n')  # 写入CSV文件的标题行
#
#     for _ in range(num_steps):
#         # 在主循环中调用LSTM并获取预测值
#         current_input = X_test[0][-look_back:]  # 使用测试集中最后一个样本的前 look_back 个时间步数据
#         current_input[-1, -1] = current_time  # 更新当前时间
#         current_input = np.reshape(current_input, (1, look_back, 2))  # 2表示两个特征，scaled_value和time
#         # 调用LSTM并获取预测值
#         predicted_value_normalized = model.predict(current_input)  # 获取归一化的预测值
#
#         # 反归一化
#         predicted_value = scaler.inverse_transform(np.array([[predicted_value_normalized[0, 0], current_time]]))
#         predicted_value = predicted_value[0, 0]  # 提取反归一化后的预测值
#
#         # 在这里你可以使用反归一化的预测值进行其他操作
#         print(f'Time {current_time}ms - LSTM Prediction: {predicted_value}')
#
#         # 将预测的时间和值追加到文件中
#         output_file.write(f'{current_time},{predicted_value}\n')
#
#         # 更新时间
#         current_time += 1


#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
# 你的数据
data = """0 93 112 143 81 87 87 108 105 111 99 90 99 99 80 114 101 98 112 106 107 105 112 102 106 104 80 105 100 92 95 106 78 120 110 101 102 91 100 103 1 112 102 104 112 110 133 117 98 114 104 105 114 84 81 87 110 109 107 107 96 114 79 102 94 89 85 102 90 91 103 92 87 104 103 102 92 103 94 114 1 93 96 94 112 91 97 94 98 97 87 82 114 98 77 103 103 101 96 93 98 110 120 116 105 94 104 94 97 105 99 97 86 107 102 110 87 92 87 103 101"""

# 将字符串数据转换为列表
data_list = list(map(int, data.split()))

# 转换为NumPy数组
data_array = np.array(data_list)

# 将数据归一化到0-1范围
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_array.reshape(-1, 1))
# 创建训练数据
look_back = 5  # 设置滑动窗口的大小
X, y = [], []

for i in range(len(scaled_data) - look_back):#每循环一次，取一次样本
    X.append(scaled_data[i:(i + look_back)].flatten())#每次在x中添加滑动窗口个数据（0-5秒的数据）
    y.append(scaled_data[i + look_back])#在y中添加label，既下个时刻的数值，用作y   （第6秒的数据）

X, y = np.array(X), np.array(y)#得到多个样本

# 将数据转换成LSTM模型的输入形状 (samples, time steps, features)
X = np.reshape(X, (X.shape[0], look_back, 1))
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, input_shape=(X.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)
# 在测试集上评估模型
predicted_values = model.predict(X_test)

# 将预测值反归一化
predicted_values = scaler.inverse_transform(predicted_values)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 打印前几个预测值和实际值
for i in range(5):
    print(f'Predicted: {predicted_values[i][0]}, Actual: {y_test[i][0]}')


# 主循环
num_steps = 20  # 设定生成未来值的步数
last_data = X_test[-1].reshape((1, look_back, 1))  # 获取测试集中最后一个样本的滑动窗口数据

for _ in range(num_steps):
    # 使用模型进行预测
    predicted_value = model.predict(last_data)

    # 将预测值添加到滑动窗口中，用于生成下一个时间步的预测
    last_data = np.append(last_data[:, 1:, :], predicted_value.reshape((1, 1, 1)), axis=1)

    # 反归一化预测值
    predicted_value = scaler.inverse_transform(predicted_value)

    # 打印未来的预测值
    print(f'Future Prediction: {predicted_value[0][0]}')

# ...（之后的代码）


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 你的数据
data = """0 93 112 143 81 87 87 108 105 111 99 90 99 99 80 114 101 98 112 106 107 105 112 102 106 104 80 105 100 92 95 106 78 120 110 101 102 91 100 103 1 112 102 104 112 110 133 117 98 114 104 105 114 84 81 87 110 109 107 107 96 114 79 102 94 89 85 102 90 91 103 92 87 104 103 102 92 103 94 114 1 93 96 94 112 91 97 94 98 97 87 82 114 98 77 103 103 101 96 93 98 110 120 116 105 94 104 94 97 105 99 97 86 107 102 110 87 92 87 103 101"""

# 将字符串数据转换为列表
data_list = list(map(int, data.split()))

# 转换为NumPy数组
data_array = np.array(data_list)

# 将数据归一化到0-1范围
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_array.reshape(-1, 1))
# 创建训练数据
look_back = 5  # 设置滑动窗口的大小
X, y = [], []

for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:(i + look_back)].flatten())
    y.append(scaled_data[i + look_back])

X, y = np.array(X), np.array(y)

# 将数据转换成LSTM模型的输入形状 (samples, time steps, features)
X = np.reshape(X, (X.shape[0], look_back, 1))
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, input_shape=(X.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2)




num_steps = 100
time = 5
current_input = X_test[0][:look_back].tolist() # 使用测试集中的一个样本作为初始输入的副本
flatten_input = np.array(current_input).flatten()
#print('current_input',current_input)


for _ in range(num_steps):
    current_input_reshaped = np.reshape(flatten_input, (1, look_back,1))
    #current_input_reshaped = np.reshape(current_input, (1, look_back, 1))#样本数，time step，特征数
    # 预测下一个时间步的值
    predicted_value_normalized = model.predict(current_input_reshaped)
    flatten_predicted = np.array(predicted_value_normalized).flatten()
    flatten_input = flatten_input[1:]#移除input的第一个元素
    flatten_input = np.concatenate([flatten_input,flatten_predicted])  # 将新的预测值添加到列表末尾
    # 将预测值反归一化
    predictions = scaler.inverse_transform(np.array(predicted_value_normalized).reshape(-1, 1))
    print(f'Step {_}: {predictions[0]}')





