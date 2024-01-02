import numpy as np
from action_space import action_space_possibility


#这里定义了一些具有固定值的全局变量
global time
time = 1  #ms

'''
各user与3个BS的分别的距离
matrix  BS0  BS1  BS2
user1
user2
user3
user4
user5
'''
#user1距离三个BS的距离，是一个1*3的向量，分别表示与BS0，BS1，BS2的距离
global dis_user1_BS
dis_user1_BS = [897.0, 265.0, 735.0]#只能连接BS1，BS2

#user2距离三个BS的距离，是一个1*3的向量，分别表示与BS0，BS1，BS2的距离
global dis_user2_BS
dis_user2_BS = [263.0, 378.0, 919.0 ]#只能连接BS0，BS1

#user3距离三个BS的距离，是一个1*3的向量，分别表示与BS0，BS1，BS2的距离
global dis_user3_BS
dis_user3_BS = [577.0, 1006.0, 461.0]#只能连接BS0，BS2

#user4距离三个BS的距离，是一个1*3的向量，分别表示与BS0，BS1，BS2的距离
global dis_user4_BS
dis_user4_BS = [1092.0, 912.0, 180.0]#只能连接BS2

#user5距离三个BS的距离，是一个1*3的向量，分别表示与BS0，BS1，BS2的距离
global dis_user5_BS
dis_user5_BS = [566.0, 583.0, 583.0]#只能连接BS0，BS1，BS2

distance_matrix = np.array([dis_user1_BS, dis_user2_BS, dis_user3_BS, dis_user4_BS, dis_user5_BS])

'''
BS在不同model下的消耗功率
分为
1.power_trans_full：activate model下处理数据
2.power_idle：activate model下没有处理数据（还没来得及转换模式，但又没有数据时）
3.power_SM1：SM1 model
4.power_SM2：SM2 model
5.power_SM3：SM3 model
'''
global power_trans_full_green
power_trans_full_green = 735

global power_trans_full_black
power_trans_full_black = 490

global power_trans_full_red
power_trans_full_red = 294

global power_idle_green
power_idle_green = 490

global power_idle_black
power_idle_black = 328

global power_idle_red
power_idle_red = 197

global power_SM1
power_SM1=157

global power_SM2
power_SM2=42.9

global power_SM3
power_SM3=28.5

'''
BS不同 work model间转换需要的时间
1.delay_Active_SM1: activate向SM1转换需要的时间
2.delay_SM1_SM2： SM1向SM2转换需要的时间
3.delay_SM2_SM3：SM2向SM3转换需要的时间
注意：单位都是 毫秒(ms)
'''
global delay_Active_SM1
delay_Active_SM1=0.0355

global delay_SM1_SM2
delay_SM1_SM2=0.5

global delay_SM2_SM3
delay_SM2_SM3=5

'''
BS work model间的转换为两个方向：
1.Activate-SM1-SM2-SM3
2.SM3-SM2-SM1-Activate
在第一个方向(Activate-SM1-SM2-SM3—Activate)的转换过程中，SM1和SM2必须有足够的停留时间才能向下一个work model转换，而第二个方向不需要
1.hold_time_SM1：在SM1model必须停留够0.07ms，才能转向SM2model
2.hold_time_SM2：在SM2model必须停留够1ms，才能转向SM3model
3.mode_time：在相同的worl model停留的时间
注意：单位都是 毫秒(ms)
'''
global hold_time_SM1
hold_time_SM1=0.07

global hold_time_SM2
hold_time_SM2=1

global mode_time
mode_time=0

'''
start_trans_no：用于标记未被处理完的数据，当当前数据被处理完后，start_trans_no+
start_trans_no0,start_trans_no1,start_trans_no2分别表示三个基站的标志
'''
#TODO,这里做了修改，将start_trans_no0/1/2改成了具体那个数据
#BS0
global start_trans_no0_2
start_trans_no0_2 = 0
global start_trans_no0_3
start_trans_no0_3=0
global start_trans_no0_5
start_trans_no0_5=0
#BS1
global start_trans_no1_1
start_trans_no1_1=0
global start_trans_no1_2
start_trans_no1_2=0
global start_trans_no1_5
start_trans_no1_5=0

#BS2
global start_trans_no2_1
start_trans_no2_1=0
global start_trans_no2_3
start_trans_no2_3=0
global start_trans_no2_4
start_trans_no2_4=0
global start_trans_no2_5
start_trans_no2_5=0


'''
三个BS各自有三个圆，表示三个不同大小的覆盖范围
1.red：半径为300
2.black：半径为500
3.green：半径为750
'''
global red_radius_small
red_radius_small = 300

global red_radius_mid
red_radius_mid = 500

global red_radius_large
red_radius_large = 750

'''
信道参数
'''
global Bandwidth
Bandwidth = 20e6

global alpha
alpha = 2.5

'''
三个基站在某个模式下的持续时间
'''
global mode_time0
mode_time0 = 0

global mode_time1
mode_time1 = 0

global mode_time2
mode_time2 = 0


'''
LSTM
'''
global look_back
look_back = 5

global look_back_roll#记录look_back向下滑动的步数
look_back_roll = 0

'''
动作空间的矩阵
'''
global action_space_matrix
action_space_matrix = action_space_possibility

'''
记录在一条命中的运行次数，用于取预测预测时间时间表中的行index,分别记录前5ms的和LSTM pre的
'''
global counter_1_for_previous_5
counter_1_for_previous_5 = 0

global counter_2_for_previous_5
counter_2_for_previous_5 = 0

global counter_3_for_previous_5
counter_3_for_previous_5 = 0

global counter_4_for_previous_5
counter_4_for_previous_5 = 0

global counter_5_for_previous_5
counter_5_for_previous_5 = 0

###for LSTM
global counter_1_for_LSTM
counter_1_for_LSTM = 0

global counter_2_for_LSTM
counter_2_for_LSTM = 0

global counter_3_for_LSTM
counter_3_for_LSTM = 0

global counter_4_for_LSTM
counter_4_for_LSTM = 0

global counter_5_for_LSTM
counter_5_for_LSTM = 0

