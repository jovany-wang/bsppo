#基站编号

#最终结果为具有11列的matrix，每列的意义如下：
'''
index0       index1     index2     index3           index4          index5
BS0_model  BS1_model  BS2_model  BS0_zoomingSize  BS1_zoomingSize  BS2_zoomingSize
     index6                     index7               index8
User1_connect_which_BS  User2_connect_which_BS  User3_connect_which_BS
       index9                  index10
User4_connect_which_BS  User5_connect_which_BS

BS_workmodel = (0,1,2,3)
0: activate   1:SM1   2.SM2   3.SM3

BS_zoomingSize = (1,2,3)
1:red circle(small)   2:black circle(middle)   3:green circle(large)


User_connect_which_BS = (0,1,2)
0: 当前用户连接的是BS0   1:当前用户连接BS1   2:当前用户连接BS2
'''

from itertools import permutations
from itertools import product
import itertools
import numpy as np
BS = [0,1,2]

#用户连接的matrix
# #连接0个用户
User_matrix1 = np.zeros([1,5])
for num in range(0,1):
    if num == 0:#连接0个用户
        User_matrix1[0,:] = -1#-1表示空值，无连接
#print('User_matrix1',User_matrix1)
#
#
# #连接1个用户
User_matrix2 = np.full([15,5],-1)
for i in range(5):
    User_matrix2[i][i] = BS[0]
for i in range(5,10):
    User_matrix2[i][i-5] = BS[1]
for i in range(10,15):
    User_matrix2[i][i-10] = BS[2]
#print('User_matrix2,',User_matrix2)

# 连接2个用户
# 1.不同基站连接2个
# 2.相同基站连接2个
# 1.
#
BS20 = [0,0,-1,-1,-1]#一个基站连接2个用户
BS21 = [1,1,-1,-1,-1]
BS22 = [2,2,-1,-1,-1]
BS210 = [-1,-1,-1,0,1]#两个个基站连接2个用户
BS211 = [-1,-1,-1,0,2]
BS212 = [-1,-1,-1,1,2]
matrix20 = []
matrix21 = []
matrix22 = []
matrix210 = []
matrix211 = []
matrix212 = []
unique20 = list(set(permutations(BS20)))
for perm in unique20:
    matrix20.append(list(perm))
# print(matrix20)
# print(len(matrix20))

unique21 = list(set(permutations(BS21)))
for perm in unique21:
    matrix21.append(list(perm))
# print(matrix21)
# print(len(matrix21))

unique22 = list(set(permutations(BS22)))
for perm in unique22:
    matrix22.append(list(perm))
# print(matrix22)
# print(len(matrix22))

unique210 = list(set(permutations(BS210)))
for perm in unique210:
    matrix210.append(list(perm))
# print(matrix210)
# print(len(matrix210))

unique211 = list(set(permutations(BS211)))
for perm in unique211:
    matrix211.append(list(perm))
# print(matrix211)
# print(len(matrix211))

unique212 = list(set(permutations(BS212)))
for perm in unique212:
    matrix212.append(list(perm))
# print(matrix212)
# print(len(matrix212))

User_matrix3 = np.concatenate((matrix20,matrix21,matrix22,matrix210,matrix211,matrix212),axis = 0)
# print('User_matrix3',User_matrix3)
# print('User_matrix3个数',len(User_matrix3))
# unique_rows, indices, counts = np.unique(User_matrix3, axis=0, return_counts=True, return_index=True)
# duplicate_rows = unique_rows[counts > 1]
#
# if len(duplicate_rows) > 0:
#     print("矩阵中存在相同的行:")
#     print(duplicate_rows)
# else:
#     print("矩阵中不存在相同的行.")

#连接3个用户
#1.一个基站连接3个用户
#2.有顺序的一个基站连接2个，一个基站连接1个
#3.三个用户分别连接3个基站
#1.
BS30 = [0,0,0,-1,-1]#3个用户同时连接基站0
BS31 = [1,1,1,-1,-1]#3个用户同时连接基站1
BS32 = [2,2,2,-1,-1]#3个用户同时连接基站2
#2.
BS3001 = [0,0,1,-1,-1]#2个用户同时连接基站0,1个用户连接基站1
BS3002 = [0,0,2,-1,-1]#2个用户同时连接基站0,1个用户连接基站1
BS3110 = [1,1,0,-1,-1]#2个用户同时连接基站0,1个用户连接基站1
BS3112 = [1,1,2,-1,-1]#2个用户同时连接基站0,1个用户连接基站1
BS3221 = [2,2,1,-1,-1]#2个用户同时连接基站0,1个用户连接基站1
BS3220 = [2,2,0,-1,-1]#2个用户同时连接基站0,1个用户连接基站1
#3.
BS3012 = [0,1,2,-1,-1]#3个用户连接3个不同基站
matrix30 = []
matrix31 = []
matrix32 = []
matrix3001 = []
matrix3002 = []
matrix3110 = []
matrix3112 = []
matrix3221 = []
matrix3220 = []
matrix3012 = []


unique30 = list(set(permutations(BS30)))
for per in unique30:
    matrix30.append(list(per))
# print(matrix30)
# print(len(matrix30))

unique31 = list(set(permutations(BS31)))
for per in unique31:
    matrix31.append(list(per))
# print(matrix31)
# print(len(matrix31))

unique32 = list(set(permutations(BS32)))
for per in unique32:
    matrix32.append(list(per))
# print(matrix32)
# print(len(matrix32))

unique3001 = list(set(permutations(BS3001)))
for per in unique3001:
    matrix3001.append(list(per))
# print(matrix3001)
# print(len(matrix3001))

unique3002 = list(set(permutations(BS3002)))
for per in unique3002:
    matrix3002.append(list(per))
# print(matrix3002)
# print(len(matrix3002))

unique3110 = list(set(permutations(BS3110)))
for per in unique3110:
    matrix3110.append(list(per))
# print(matrix3110)
# print(len(matrix3110))

unique3112 = list(set(permutations(BS3112)))
for per in unique3112:
    matrix3112.append(list(per))
# print(matrix3112)
# print(len(matrix3112))

unique3221 = list(set(permutations(BS3221)))
for per in unique3221:
    matrix3221.append(list(per))
# print(matrix3221)
# print(len(matrix3221))

unique3220 = list(set(permutations(BS3220)))
for per in unique3220:
    matrix3220.append(list(per))
# print(matrix3220)
# print(len(matrix3220))

unique3012 = list(set(permutations(BS3012)))
for per in unique3012:
    matrix3012.append(list(per))
# print(matrix3012)
# print(len(matrix3012))


User_matrix4 = np.concatenate((matrix30,matrix31,matrix32,matrix3001,matrix3002,matrix3110,matrix3112,matrix3221,matrix3220,matrix3012),axis = 0)
# print('User_matrix3',User_matrix3)
# print('User_matrix3个数',len(User_matrix3))
# unique_rows, indices, counts = np.unique(User_matrix3, axis=0, return_counts=True, return_index=True)
# duplicate_rows = unique_rows[counts > 1]
#
# if len(duplicate_rows) > 0:
#     print("矩阵中存在相同的行:")
#     print(duplicate_rows)
# else:
#     print("矩阵中不存在相同的行.")

#连接4个用户
#1.4个用户在同一基站中（连接1个基站）
#2.分1，3， 1个user在一个基站，其余3个user在同一个基站（连接2个基站）
#3.分2，2， 2个user在同一个基站，其余2个user在同一个基站 （连接2个基站）
#4.分2，1，1 2个user在同一个基站，其余2个user在两个个基站 （连接3个基站）
#
# #1.
BS40 = [0,0,0,0,-1]
BS41 = [1,1,1,1,-1]
BS42 = [2,2,2,2,-1]
#2.3-1
BS40001 = [0,0,0,1,-1]
BS40002 = [0,0,0,2,-1]
BS41110 = [1,1,1,0,-1]
BS41112 = [1,1,1,2,-1]
BS42220 = [2,2,2,0,-1]
BS42221 = [2,2,2,1,-1]
#3. 2-2
BS41122 = [1,1,2,2,-1]
BS41100 = [1,1,0,0,-1]
BS40022 = [0,0,2,2,-1]
#4.
BS40012 = [0,0,1,2,-1]
BS41102 = [1,1,0,2,-1]
BS42201 = [2,2,1,0,-1]

matrix40 = []
matrix41 = []
matrix42 = []

matrix40001 = []
matrix40002 = []
matrix41110 = []
matrix41112 = []
matrix42220 = []
matrix42221 = []

matrix41100 = []
matrix41122 = []
matrix40022 = []

matrix40012 = []
matrix41102 = []
matrix42201 = []

unique40 = list(set(permutations(BS40)))
for per in unique40:
    matrix40.append(list(per))
# print(matrix40)
# print(len(matrix40))

unique41 = list(set(permutations(BS41)))
for per in unique41:
    matrix41.append(list(per))
# print(matrix41)
# print(len(matrix41))

unique42 = list(set(permutations(BS42)))
for per in unique42:
    matrix42.append(list(per))
# print(matrix42)
# print(len(matrix42))

unique40001 = list(set(permutations(BS40001)))
for per in unique40001:
    matrix40001.append(list(per))
# print(matrix40001)
# print(len(matrix40001))

unique40002 = list(set(permutations(BS40002)))
for per in unique40002:
    matrix40002.append(list(per))
# print(matrix40002)
# print(len(matrix40002))

unique41110 = list(set(permutations(BS41110)))
for per in unique41110:
    matrix41110.append(list(per))
# print(matrix41110)
# print(len(matrix41110))

unique41112 = list(set(permutations(BS41112)))
for per in unique41112:
    matrix41112.append(list(per))
# print(matrix41112)
# print(len(matrix41112))

unique42220 = list(set(permutations(BS42220)))
for per in unique42220:
    matrix42220.append(list(per))
# print(matrix42220)
# print(len(matrix42220))

unique42221 = list(set(permutations(BS42221)))
for per in unique42221:
    matrix42221.append(list(per))
# print(matrix42221)
# print(len(matrix42221))

unique41100 = list(set(permutations(BS41100)))
for per in unique41100:
    matrix41100.append(list(per))
# print(matrix41100)
# print(len(matrix41100))

unique41122 = list(set(permutations(BS41122)))
for per in unique41122:
    matrix41122.append(list(per))
# print(matrix41122)
# print(len(matrix41122))

unique40022 = list(set(permutations(BS40022)))
for per in unique40022:
    matrix40022.append(list(per))
# print(matrix40022)
# print(len(matrix40022))

unique40012 = list(set(permutations(BS40012)))
for per in unique40012:
    matrix40012.append(list(per))
# print(matrix40012)
# print(len(matrix40012))

unique41102 = list(set(permutations(BS41102)))
for per in unique41102:
    matrix41102.append(list(per))
# print(matrix41102)
# print(len(matrix41102))

unique42201 = list(set(permutations(BS42201)))
for per in unique42201:
    matrix42201.append(list(per))
# print(matrix42201)
# print(len(matrix42201))


User_matrix5 = np.concatenate((matrix40,matrix41,matrix42,matrix40001,matrix40002,matrix41110,matrix41112,matrix42220,matrix42221,matrix41100,matrix41122,matrix40022,matrix40012,matrix41102,matrix42201),axis = 0)
# print('User_matrix3',User_matrix4)
# print('User_matrix3个数',len(User_matrix4))
# unique_rows, indices, counts = np.unique(User_matrix4, axis=0, return_counts=True, return_index=True)
# duplicate_rows = unique_rows[counts > 1]
#
# if len(duplicate_rows) > 0:
#     print("矩阵中存在相同的行:")
#     print(duplicate_rows)
# else:
#     print("矩阵中不存在相同的行.")

#连接5个用户
#1.同一个基站连接5个用户（不合法）
#2.连接2个基站：4-1和3-2，其中4-1只有BS可以实现4

#同一基站
BS50 = [0,0,0,0,0]
BS51 = [1,1,1,1,1]
BS52 = [2,2,2,2,2]
#4-1
BS500001 = [0,0,0,0,1]
BS500002 = [0,0,0,0,2]
BS511110 = [1,1,1,1,0]
BS511112 = [1,1,1,1,2]
BS522220 = [2,2,2,2,0]
BS522221 = [2,2,2,2,1]
#3-2(2个基站)
BS500011 = [0,0,0,1,1]
BS500022 = [0,0,0,2,2]
BS511100 = [1,1,1,0,0]
BS511122 = [1,1,1,2,2]
BS522200 = [2,2,2,0,0]
BS522211 = [2,2,2,1,1]
#3个基站  3-1-1
BS500012 = [0,0,0,1,2]
BS511102 = [1,1,1,0,2]
BS522201 = [2,2,2,0,1]
#3个基站  2-2-1
BS500112 = [0,0,1,1,2]
BS500221 = [0,0,2,2,1]
BS522110 = [2,2,1,1,0]

matrix50 = []
matrix51 = []
matrix52 = []
matrix500001 = []
matrix500002 = []
matrix511110 = []
matrix511112 = []
matrix522220 = []
matrix522221 = []

matrix500011 = []
matrix500022 = []
matrix511100 = []
matrix511122 = []
matrix522200 = []
matrix522211 = []

matrix500012 = []
matrix511102 = []
matrix522201 = []

matrix500112 = []
matrix500221 = []
matrix522110 = []

#1
unique50 = list(set(permutations(BS50)))
for per in unique50:
    matrix50.append(list(per))
# print(matrix50)
# print(len(matrix50))

unique51 = list(set(permutations(BS51)))
for per in unique51:
    matrix51.append(list(per))
# print(matrix51)
# print(len(matrix51))

unique52 = list(set(permutations(BS52)))
for per in unique52:
    matrix52.append(list(per))
# print(matrix52)
# print(len(matrix52))
#2
unique500001 = list(set(permutations(BS500001)))
for per in unique500001:
    matrix500001.append(list(per))
# print(matrix500001)
# print(len(matrix500001))

unique500002 = list(set(permutations(BS500002)))
for per in unique500002:
    matrix500002.append(list(per))
# print(matrix500002)
# print(len(matrix500002))

unique511110 = list(set(permutations(BS511110)))
for per in unique511110:
    matrix511110.append(list(per))
# print(matrix511110)
# print(len(matrix511110))

unique511112 = list(set(permutations(BS511112)))
for per in unique511112:
    matrix511112.append(list(per))
# print(matrix511112)
# print(len(matrix511112))

unique522220 = list(set(permutations(BS522220)))
for per in unique522220:
    matrix522220.append(list(per))
# print(matrix522220)
# print(len(matrix522220))

unique522221 = list(set(permutations(BS522221)))
for per in unique522221:
    matrix522221.append(list(per))
# print(matrix522221)
# print(len(matrix522221))
#3
unique500011 = list(set(permutations(BS500011)))
for per in unique500011:
    matrix500011.append(list(per))
# print(matrix500011)
# print(len(matrix500011))

unique500022 = list(set(permutations(BS500022)))
for per in unique500022:
    matrix500022.append(list(per))
# print(matrix500022)
# print(len(matrix500022))

unique511100 = list(set(permutations(BS511100)))
for per in unique511100:
    matrix511100.append(list(per))
# print(matrix511100)
# print(len(matrix511100))

unique511122 = list(set(permutations(BS511122)))
for per in unique511122:
    matrix511122.append(list(per))
# print(matrix511122)
# print(len(matrix511122))

unique522200 = list(set(permutations(BS522200)))
for per in unique522200:
    matrix522200.append(list(per))
# print(matrix522200)
# print(len(matrix522200))

unique522211 = list(set(permutations(BS522211)))
for per in unique522211:
    matrix522211.append(list(per))
# print(matrix522211)
# print(len(matrix522211))
#4
unique500012 = list(set(permutations(BS500012)))
for per in unique500012:
    matrix500012.append(list(per))
# print(matrix500012)
# print(len(matrix500012))

unique511102 = list(set(permutations(BS511102)))
for per in unique511102:
    matrix511102.append(list(per))
# print(matrix511102)
# print(len(matrix511102))

unique522201 = list(set(permutations(BS522201)))
for per in unique522201:
    matrix522201.append(list(per))
# print(matrix522201)
# print(len(matrix522201))
#5
unique500112 = list(set(permutations(BS500112)))
for per in unique500112:
    matrix500112.append(list(per))
# print(matrix500112)
# print(len(matrix500112))

unique500221 = list(set(permutations(BS500221)))
for per in unique500221:
    matrix500221.append(list(per))
# print(matrix500221)
# print(len(matrix500221))


unique522110 = list(set(permutations(BS522110)))
for per in unique522110:
    matrix522110.append(list(per))
# print(matrix522110)
# print(len(matrix522110))


User_matrix6 = np.concatenate((matrix50,matrix51,matrix52,matrix500001,matrix500002,matrix511110,matrix511112,matrix522220,matrix522221,matrix500011,matrix500022,matrix511100,matrix511122,matrix522200,matrix522211,matrix500012,matrix511102,matrix522201,matrix500112,matrix500221,matrix522110),axis = 0)
# print('User_matrix5',User_matrix5)
# print('User_matrix5个数',len(User_matrix5))
#所有的user-BS连接的matrix（5列）
User_connect_action = np.concatenate((User_matrix1,User_matrix2,User_matrix3,User_matrix4,User_matrix5,User_matrix6),axis = 0)
# print('User_connect_action',User_connect_action)
# print('User_connect_action的行数',len(User_connect_action))

# unique_rows, indices, counts = np.unique(User_connect_action, axis=0, return_counts=True, return_index=True)
# duplicate_rows = unique_rows[counts > 1]
# if len(duplicate_rows) > 0:
#     print()
#     print("矩阵中存在相同的行:")
#     print(duplicate_rows)
# else:
#     print("矩阵中不存在相同的行.")


#开始加判断条件，去掉不合法的情况：
#先根据数量去掉肯定不存在的情况
#再根据连接范围去掉基站够不到的情况


#还没有将user-BS的连接与BSModel和zooming 的matrix拼接到一起，先对user—BS连接去掉不合法的情况
#删除同一基站连接5用户的情况
remove_5user = 5
#删除BS0,BS1连接4用户的情况
remove_5user = 4
#初始化要删除行的index
remove_row_index = []
for i,row in enumerate(User_connect_action):# i为index row为当前行元素
    if np.sum(row == 0) == 5 or np.sum(row == 1) == 5 or np.sum(row == 2) == 5 or np.sum(row == 0) == 4 or np.sum(row == 1) == 4:
        remove_row_index.append(i)

# print('new1删除的行',User_connect_action[remove_row_index])
new_User_connect_action = np.delete(User_connect_action, remove_row_index, axis=0)
# print('new_User_connect_action',new_User_connect_action)
# print('剩余行数new',len(new_User_connect_action))

#分别按照BS0，BS1，BS2这个顺序减去覆盖范围内不可能连接的情况
#BS0：删除第0列和第3列出现0的行
remove_row_BS0_index = []
for i, row in enumerate(new_User_connect_action):
    if row[0] == 0 or row[3] == 0:
        remove_row_BS0_index.append(i)
# print('new2删除的行',new_User_connect_action[remove_row_BS0_index])
new2_User_connect_action = np.delete(new_User_connect_action, remove_row_BS0_index, axis=0)
# print('new2_User_connect_action',new2_User_connect_action)
# print('剩余行数new2',len(new2_User_connect_action))

#BS1：删除第0列和第3列出现0的行
remove_row_BS1_index = []
for i, row in enumerate(new2_User_connect_action):
    if row[2] == 1 or row[3] == 1:
        remove_row_BS1_index.append(i)
# print('new3删除的行',new2_User_connect_action[remove_row_BS1_index])
new3_User_connect_action = np.delete(new2_User_connect_action, remove_row_BS1_index, axis=0)
# print('new3_User_connect_action',new3_User_connect_action)
# print('new3剩余行数',len(new3_User_connect_action))

##BS2：删除第0列和第3列出现0的行
remove_row_BS2_index = []
for i, row in enumerate(new3_User_connect_action):
    if row[1] == 2:
        remove_row_BS2_index.append(i)
# print('new4删除的行',new3_User_connect_action[remove_row_BS2_index])
new4_User_connect_action = np.delete(new3_User_connect_action, remove_row_BS2_index, axis=0)
# print('new4_User_connect_action',new4_User_connect_action)
# print('new4剩余行数',len(new4_User_connect_action))

User_con_action = new4_User_connect_action
#print('User_con_action行数为',len(User_con_action))


#model 与 zooming
#1.model：每个基站有4中工作模式
BS_model = [0, 1, 2, 3]
matrix1 = list(product(BS_model, repeat=3))
# print(matrix1)
# print(len(matrix1))

#2.每个基站有3中zooming
BS_zooming = [0,1,2,3]#这里做了修改，加了0，0是在BSmodel为SM时的情况
matrix2 = list(product(BS_zooming, repeat=3))
# print(matrix2)
# print(len(matrix2))
tiled_matrix1 = np.tile(matrix2, (len(matrix1), 1))
tiled_matrix2 = np.repeat(matrix1, len(matrix2), axis=0)
# 交换拼接的顺序
merged_matrix = np.concatenate((tiled_matrix2, tiled_matrix1), axis=1)

# print(merged_matrix)
# print(len(merged_matrix))
model_zooming = merged_matrix

# unique_rows, indices, counts = np.unique(model_zooming, axis=0, return_counts=True, return_index=True)
# duplicate_rows = unique_rows[counts > 1]

# if len(duplicate_rows) > 0:
#     print()
#     print("矩阵中存在相同的行:")
#     print(duplicate_rows)
# else:
#     print("矩阵中不存在相同的行.")



#去掉model_zooming中不合法的情况
#首先去掉工作模式不为0时，对应的zooming数值不为2的情况
remove_model_zooming_index = []
for i, row in enumerate(model_zooming):
    if row[0] != 0 and row[3] != 0 or row[1] != 0 and row[4] != 0 or row[2] != 0 and row[5] != 0:#将BSmodel为SM1，SM2，SM3时的zooming设置为0
        remove_model_zooming_index.append(i)
    elif row[0] == 0 and row[3] == 0 or row[1] == 0 and row[4] == 0 or row[2] == 0 and row[5] == 0:#当BSmodel为0时，说明是activate的，Zooming必须有大小，不能为0
        remove_model_zooming_index.append(i)
# print('model_zooming中删除的行为：',model_zooming[remove_model_zooming_index])
model_zooming_action = np.delete(model_zooming,remove_model_zooming_index,axis=0)
# print('model_zooming_update', model_zooming_update)
#print('model_zooming_action行数为', len(model_zooming_action))#144行

#将User_con_action与model_zooming_action  merged

matrix11 = model_zooming_action
matrix22 = User_con_action
tiled_matrix11 = np.tile(matrix22, (len(matrix11), 1))
tiled_matrix22 = np.repeat(matrix11, len(matrix22), axis=0)
# 交换拼接的顺序
action_space_temp = np.concatenate((tiled_matrix22, tiled_matrix11), axis=1)
# print(action_space_temp )
# print('action_space_temp行数',len(action_space_temp))
# print('action_space_temp列数',action_space_temp.shape[1])


#在action_space_temp 中再去掉不合法的情况
#根据连接范围
#判断BS0,先判断是否在工作模式，如果是，则看他的zooming大小，去掉没有覆盖的范围
#                          如果不是，他的zooming2，则直接去掉不在2范围内的
remove_BS0_action_space = []
for i, row in enumerate(action_space_temp):
    if row[0] != 0 and row[7] == 0 or row[0] != 0 and row[8] == 0 or row[0] != 0 and row[10] == 0:#去掉基站模式为SM时，有用户连接的情况
        remove_BS0_action_space.append(i)
    elif row[0] == 0:#为0时，还要判断他的当前zooming大小
        if row[3] == 1 and row[8] == 0 or row[3] == 1 and row[10] == 0:
            remove_BS0_action_space.append(i)
        elif row[3] == 2 and row[8] == 0 or row[3] == 2 and row[10] == 0:
            remove_BS0_action_space.append(i)
# print('action_space_temp删除的行',action_space_temp[remove_BS0_action_space])
action_space_temp2 = np.delete(action_space_temp, remove_BS0_action_space, axis=0)
# print('action_space_temp2',action_space_temp2)
# print('action_space_temp2剩余行数',len(action_space_temp2))

#BS1
remove_BS1_action_space = []
for i, row in enumerate(action_space_temp2):
    if row[1] != 0 and row[6] == 1 or row[1] != 0  and row[7] == 1 or row[1] != 0  and row[10] == 1:
        remove_BS1_action_space.append(i)
    elif row[1] == 0:#为0时，还要判断他的当前zooming大小
        if row[4] == 1 and row[7] == 1 or row[4] == 1 and row[10] == 1:
            remove_BS1_action_space.append(i)
        elif row[4] == 2 and row[7] == 1 or row[4] == 2 and row[10] == 1:
            remove_BS1_action_space.append(i)
# print('action_space_temp删除的行',action_space_temp2[remove_BS1_action_space])
action_space_temp3 = np.delete(action_space_temp2, remove_BS1_action_space, axis=0)
# print('action_space_temp3',action_space_temp3)
# print('action_space_temp3剩余行数',len(action_space_temp3))


remove_BS3_action_space = []
for i, row in enumerate(action_space_temp3):
    if row[2] != 0 and row[8] == 2 or row[2] != 0 and row[9] == 2 or row[2] != 0 and row[10] == 2:
        remove_BS3_action_space.append(i)
    elif row[1] == 0:#为0时，还要判断他的当前zooming大小
        if row[5] == 1 and row[8] == 2 or row[5] == 1 and row[10] == 2 or row[5] == 1 and row[6] == 2:
            remove_BS3_action_space.append(i)
        elif row[5] == 2 and row[10] == 0 or row[5] == 2 and row[6] == 0:
            remove_BS3_action_space.append(i)
# print('action_space_temp删除的行',action_space_temp3[remove_BS3_action_space])
action_space_temp4 = np.delete(action_space_temp3, remove_BS3_action_space, axis=0)
# print('action_space_temp4',action_space_temp4)
# print('action_space_temp4剩余行数',len(action_space_temp4))

action_space_possibility = action_space_temp4 #7031
print('action_space_possibility,',action_space_possibility)
print('action_space_possibility,',len(action_space_possibility))#5136










































