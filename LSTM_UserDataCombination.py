import scipy.io
import pandas as pd

#User1
# 读取.mat文件
annot1 = scipy.io.loadmat('generationTime1.mat')
column1_data = annot1['generationTime']
annot2 = scipy.io.loadmat('packetSizes1.mat')
column2_data = annot2['packetSizes']
# 将数据放入DataFrame
df = pd.DataFrame({
    'generationTime': column1_data.flatten(),
    'packetSizes': column2_data.flatten()
})
# 保存到同一个.csv文件
df.to_csv('user1_data.csv', index=False)


#User2
# 读取第一个.mat文件
annot21 = scipy.io.loadmat('generationTime2.mat')
column1_data21 = annot21['generationTime']
# 读取第二个.mat文件
annot22 = scipy.io.loadmat('packedSizes2.mat')
column2_data22 = annot22['packetSizes']
# 将数据放入DataFrame
df = pd.DataFrame({
    'generationTime': column1_data21.flatten(),
    'packetSizes': column2_data22.flatten()
})
# 保存到同一个.csv文件
df.to_csv('user2_data.csv', index=False)



#User3,采用了user2的数据，直接换generationTime.mat和packetSizes1.mat两个数据机就可以
# 读取第一个.mat文件
annot31 = scipy.io.loadmat('generationTime1.mat')
column1_data3 = annot31['generationTime']
# column1_data = mat_file1['column_name']  # 请替换 'column_name' 为实际的列名
# 读取第二个.mat文件
annot32 = scipy.io.loadmat('packetSizes1.mat')
column2_data3 = annot32['packetSizes']
# column2_data = mat_file2['column_name']  # 请替换 'column_name' 为实际的列名
# 将数据放入DataFrame
df = pd.DataFrame({
    'generationTime': column1_data3.flatten(),
    'packetSizes': column2_data3.flatten()
})
# 保存到同一个.csv文件
df.to_csv('user3_data.csv', index=False)

#User4
# 读取第一个.mat文件
annot41 = scipy.io.loadmat('generationTime1.mat')
column1_data4 = annot1['generationTime']
# 读取第二个.mat文件
annot42 = scipy.io.loadmat('packetSizes1.mat')
column2_data4 = annot2['packetSizes']
# 将数据放入DataFrame
df = pd.DataFrame({
    'generationTime': column1_data4.flatten(),
    'packetSizes': column2_data4.flatten()
})
# 保存到同一个.csv文件
df.to_csv('user4_data.csv', index=False)


#User5
# 读取第一个.mat文件
annot51 = scipy.io.loadmat('generationTime1.mat')
column1_data5 = annot1['generationTime']
# 读取第二个.mat文件
annot52 = scipy.io.loadmat('packetSizes1.mat')
column2_data5 = annot52['packetSizes']
# 将数据放入DataFrame
df = pd.DataFrame({
    'generationTime': column1_data5.flatten(),
    'packetSizes': column2_data5.flatten()
})
# 保存到同一个.csv文件
df.to_csv('user5_data.csv', index=False)
