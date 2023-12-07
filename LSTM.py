import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LambdaCallback

# 读取CSV文件
file_path = 'user1_LSTMDate.csv'  # 请替换为你的CSV文件路径
df = pd.read_csv(file_path, header=None, names=['value'])

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_value'] = scaler.fit_transform(df['value'].values.reshape(-1, 1))

# 添加时间列
df['time'] = np.arange(0, len(df))

# 创建训练数据
look_back = 100  # 设置时间步长，可以根据实际情况调整
X, y, time = [], [], []

for i in range(len(df) - look_back):
    X.append(df[['scaled_value', 'time']][i:(i + look_back)].values)
    y.append(df['scaled_value'][i + look_back])
    time.append(df['time'][i + look_back])

X, y, time = np.array(X), np.array(y), np.array(time)

# 将数据转换成LSTM模型的输入形状 (samples, time steps, features)
X = np.reshape(X, (X.shape[0], look_back, 2))  # 2表示两个特征，scaled_value和time

# 划分训练集和测试集
X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(X, y, time, test_size=0.2, random_state=42)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(units=1, activation='relu'))
model.compile(optimizer='adam', loss='mean_squared_error')

# 定义一个回调函数来在每个周期结束时打印下一时刻的预测值
def print_next_prediction(epoch, logs):
    test_input = X_test[0][-look_back:]  # 使用测试集中最后一个样本的前 look_back 个时间步数据
    test_input = np.reshape(test_input, (1, look_back, 2))  # 2表示两个特征，scaled_value和time
    predicted_value = np.round(model.predict(test_input))  # 四舍五入到最近的整数
    true_value = y_test[0]
    current_time = time_test[0]
    print(f'Epoch {epoch + 1} - Next Prediction at Time {current_time + 1}ms: {predicted_value[0][0]}, True Value: {true_value}')

prediction_callback = LambdaCallback(on_epoch_end=print_next_prediction)

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2, callbacks=[prediction_callback])

# 在测试集上评估模型
loss = model.evaluate(X_test, y_test, verbose=2)
print(f'Mean Squared Error on Test Set: {loss}')

# 主循环
num_steps = 10  # 设定主循环的总步数
current_time = time_test[0] + 1  # 从第31ms开始

for _ in range(num_steps):
    # 在主循环中调用LSTM并获取预测值
    current_input = X_test[0][-look_back:]  # 使用测试集中最后一个样本的前 look_back 个时间步数据
    current_input[-1, -1] = current_time  # 更新当前时间
    current_input = np.reshape(current_input, (1, look_back, 2))  # 2表示两个特征，scaled_value和time

    # 调用LSTM并获取预测值
    predicted_value = np.round(model.predict(current_input))  # 四舍五入到最近的整数

    # 在这里你可以使用预测值进行其他操作
    print(f'Time {current_time}ms - LSTM Prediction: {predicted_value[0][0]}')

    # 更新时间
    current_time += 1
