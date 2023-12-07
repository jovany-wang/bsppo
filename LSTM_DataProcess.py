#先把data按照time的index放入新的csv中（注意：每个时间都对应了很多值，先给他求个和）必须要进行求和，因为如果不求和，用0补全的数据代表1ms连续，而49秒内的数据是连续的好几个，这样每个data的time scale是不同的
#再找空值，用0补全

import pandas as pd
#User1
# 读取 CSV 文件
file_path = 'user1_data.csv'
df = pd.read_csv(file_path)

# 创建 DataFrame 并按 'generationTime' 列求和
data = {'generationTime': df['generationTime'].tolist(), 'packetSizes': df['packetSizes'].tolist()}
df = pd.DataFrame(data).set_index('generationTime')
grouped = df.groupby(df.index)['packetSizes'].sum().reset_index()

# 重新索引，用0填充NaN值
df2 = grouped.set_index('generationTime')
df2 = df2.reindex(range(df2.index.min(), df2.index.max()+1)).fillna(0)

# 重置索引并保存到 CSV 文件
df2 = df2.reset_index(drop = True)
output_file_path = 'user1_LSTMDate.csv'
df2.to_csv(output_file_path, index=False,header=False)


#User2
import pandas as pd

file_path = 'user2_data.csv'
df = pd.read_csv(file_path)
data = {'generationTime': df['generationTime'].tolist(), 'packetSizes': df['packetSizes'].tolist()}
df = pd.DataFrame(data).set_index('generationTime')
grouped = df.groupby(df.index)['packetSizes'].sum().reset_index()
df2 = grouped.set_index('generationTime')
df2 = df2.reindex(range(df2.index.min(), df2.index.max()+1)).fillna(0)
df2 = df2.reset_index(drop=True)
output_file_path = 'user2_LSTMDate.csv'
df2.to_csv(output_file_path, index=False, header=False)


#User3
# 读取 CSV 文件
file_path = 'user3_data.csv'
df = pd.read_csv(file_path)
data = {'generationTime': df['generationTime'].tolist(), 'packetSizes': df['packetSizes'].tolist()}
df = pd.DataFrame(data).set_index('generationTime')
grouped = df.groupby(df.index)['packetSizes'].sum().reset_index()
df2 = grouped.set_index('generationTime')
df2 = df2.reindex(range(df2.index.min(), df2.index.max()+1)).fillna(0)
df2 = df2.reset_index(drop=True)
output_file_path = 'user3_LSTMDate.csv'
df2.to_csv(output_file_path, index=False, header=False)

#User4
# 读取 CSV 文件
file_path = 'user4_data.csv'
df = pd.read_csv(file_path)
data = {'generationTime': df['generationTime'].tolist(), 'packetSizes': df['packetSizes'].tolist()}
df = pd.DataFrame(data).set_index('generationTime')
grouped = df.groupby(df.index)['packetSizes'].sum().reset_index()
df2 = grouped.set_index('generationTime')
df2 = df2.reindex(range(df2.index.min(), df2.index.max()+1)).fillna(0)
df2 = df2.reset_index(drop=True)
output_file_path = 'user4_LSTMDate.csv'
df2.to_csv(output_file_path, index=False, header=False)

#User5
# 读取 CSV 文件
file_path = 'user5_data.csv'
df = pd.read_csv(file_path)
data = {'generationTime': df['generationTime'].tolist(), 'packetSizes': df['packetSizes'].tolist()}
df = pd.DataFrame(data).set_index('generationTime')
grouped = df.groupby(df.index)['packetSizes'].sum().reset_index()
df2 = grouped.set_index('generationTime')
df2 = df2.reindex(range(df2.index.min(), df2.index.max()+1)).fillna(0)
df2 = df2.reset_index(drop=True)
output_file_path = 'user5_LSTMDate.csv'
df2.to_csv(output_file_path, index=False, header=False)
