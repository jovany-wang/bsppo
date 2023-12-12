#分别计算三个BS的到达数据并将其分别放入3个arrival_data_matrix0，arrival_data_matrix1，arrival_data_matrix2中
import numpy as np

import Vars


#写完这个再把BS power中的trans_rate替换掉
#再看main.py,再PPO的，main.py和step中将他们穿串起来
'''
在action中，action共11列，后5列表示user-BS的连接，对应的列index和user为：
    index6   index7   index8   index9   index10
     user1    user2    user3     user4    user5
action 的前6列对应
    index0   index1   index2   index3   index4   index5
    BS0mode  BS1mode  BS2mode  BS0_zom  BS1_zom  BS2_zom
其中：
    BSmode = {0:activate, 1:SM1, 2:SM2, 3:SM3}
    BS_zom = {1:red, 2:black, 3:green}
    user_BS = {0:当前user连接BS0， 1:当前user连接BS1， 2:当前user连接BS2}
'''

'''
def trans_rate_15(dis,action)返回每个user与每个BS连接时的trans_rate,每个用户有一个trans_list,分别表示与三个基站的trans_rate

1. trans_rate_list_user1: BS0,BS1,BS2
2. trans_rate_list_user2: BS0,BS1,BS2 
3. trans_rate_list_user3: BS0,BS1,BS2 
4. trans_rate_list_user4: BS0,BS1,BS2 
5. trans_rate_list_user5: BS0,BS1,BS2 
分别用trans_rate_list_user1[0],trans_rate_list_user1[1],trans_rate_list_user1[2]来获取每个user与对应BS的trans_rate
'''


#计算每个用户和每个BS之间的trans_rate(共15个)
def trans_rate_15(action):
    SNR_list_user1 = [] #for each user
    SNR_list_user2 = []
    SNR_list_user3 = []
    SNR_list_user4 = []
    SNR_list_user5 = []
    trans_rate_list_user1 = []#三元素list，分别为该用户与BS0，BS1，BS2的三条信道的trans_rate
    trans_rate_list_user2 = []
    trans_rate_list_user3 = []
    trans_rate_list_user4 = []
    trans_rate_list_user5 = []
    h_15 = []
    #user1
    for index,colu in enumerate(Vars.distance_matrix[0,:]):
        h = np.random.normal(loc=0, scale=1) / (colu**(Vars.alpha / 2))
        h_abs = np.abs(h)
        h_15.append(h_abs)
        h_squ =np.square(h_abs)
        if index == 0:#user1-BS0
            if action[3] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[3] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[3] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        elif index == 1:#user1-BS1
            if action[4] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[4] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[4] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        elif index == 2:#user1-BS2
            if action[5] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[5] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[5] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        trans_rate = Vars.Bandwidth * np.log2(1 + SNR) / 1000 #in kbpms
        SNR_list_user1.append(SNR)
        trans_rate_list_user1.append(trans_rate)
    #user2
    for row,colu in enumerate(Vars.distance_matrix[1,:]):
        h = np.random.normal(loc=0, scale=1) / (colu**(Vars.alpha / 2))
        h_abs = np.abs(h)
        h_15.append(h_abs)
        h_squ =np.square(h_abs)
        if index == 0:#user2-BS0
            if action[3] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[3] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[3] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        elif index == 1:#user2-BS1
            if action[4] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[4] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[4] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        elif index == 2:#user2-BS2
            if action[5] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[5] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[5] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        trans_rate = Vars.Bandwidth * np.log2(1 + SNR) / 1000 #in kbpms
        SNR_list_user2.append(SNR)
        trans_rate_list_user2.append(trans_rate)
    #user3
    for row,colu in enumerate(Vars.distance_matrix[2,:]):
        h = np.random.normal(loc=0, scale=1) / (colu**(Vars.alpha / 2))
        h_abs = np.abs(h)
        h_15.append(h_abs)
        h_squ =np.square(h_abs)
        if index == 0:#user3-BS0
            if action[3] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[3] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[3] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        elif index == 1:#user3-BS1
            if action[4] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[4] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[4] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        elif index == 2:#user3-BS2
            if action[5] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[5] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[5] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        trans_rate = Vars.Bandwidth * np.log2(1 + SNR) / 1000 #in kbpms
        SNR_list_user3.append(SNR)
        trans_rate_list_user3.append(trans_rate)

    #user4
    for row,colu in enumerate(Vars.distance_matrix[3,:]):
        h = np.random.normal(loc=0, scale=1) / (colu**(Vars.alpha / 2))
        h_abs = np.abs(h)
        h_15.append(h_abs)
        h_squ =np.square(h_abs)
        if index == 0:#user4-BS0
            if action[3] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[3] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[3] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        elif index == 1:#user2-BS1
            if action[4] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[4] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[4] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        elif index == 2:#user-BS2
            if action[5] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[5] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[5] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        trans_rate = Vars.Bandwidth * np.log2(1 + SNR) / 1000 #in kbpms
        SNR_list_user4.append(SNR)
        trans_rate_list_user4.append(trans_rate)
    #user5
    for row,colu in enumerate(Vars.distance_matrix[4,:]):
        h = np.random.normal(loc=0, scale=1) / (colu**(Vars.alpha / 2))
        h_abs = np.abs(h)
        h_15.append(h_abs)
        h_squ =np.square(h_abs)
        if index == 0:#user4-BS0
            if action[3] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[3] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[3] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        elif index == 1:#user2-BS1
            if action[4] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[4] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[4] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        elif index == 2:#user-BS2
            if action[5] == 1:
                SNR = Vars.power_trans_full_red * h_squ
            elif action[5] == 2:
                SNR = Vars.power_trans_full_black * h_squ
            elif action[5] == 3:
                SNR = Vars.power_trans_full_green * h_squ
        trans_rate = Vars.Bandwidth * np.log2(1 + SNR) / 1000 #in kbpms
        SNR_list_user5.append(SNR)
        trans_rate_list_user5.append(trans_rate)
    return h_15, trans_rate_list_user1,trans_rate_list_user2,trans_rate_list_user3,trans_rate_list_user4,trans_rate_list_user5
#return  h_15, trans_rate_list_user1,trans_rate_list_user2,trans_rate_list_user3,trans_rate_list_user4,trans_rate_list_user5

#根据action计算每个BS上的trans_rate
def trans_rate_3BS(action,trans_rate_list_user1,trans_rate_list_user2,trans_rate_list_user3,trans_rate_list_user4,trans_rate_list_user5):
    #获取action后5列的user-BS的连接情况
    last_five_column = action[-5:]
    #BS0
    trans_BS0= np.where(last_five_column == 0)[0]#判断那个用户与BS0连接
    trans_BS0_index = trans_BS0 + (len(action) - len(last_five_columns))#[6,7,8] 哪个user连接的
    #判断当前的BS与几个用户连接，len(trans_BS0_index)为几，就是与几个用户连接
    if len(trans_BS0_index) == 3: #user2,3,5
        trans_rate_BS0 = trans_rate_list_user2[0] + trans_rate_list_user3[0] + trans_rate_list_user5[0]
    elif len(trans_BS0_index) == 2: #user23/35/25
        if action[7] == 0 and action[8] == 0:#user2,3
            trans_rate_BS0 = trans_rate_list_user2[0] + trans_rate_list_user3[0]
        elif action[8] == 0 and action[10] == 0:#user3,5
            trans_rate_BS0 = trans_rate_list_user3[0] + trans_rate_list_user5[0]
        elif action[7] == 0 and action[10] == 0:#user2,5
            trans_rate_BS0 = trans_rate_list_user7[0] + trans_rate_list_user5[0]
    elif len(trans_BS0_index) == 1:#只与一个user连接，2/3/5
        if action[7] == 0:
            trans_rate_BS0 = trans_rate_list_user2[0]
        elif action[8] == 0:
            trans_rate_BS0 = trans_rate_list_user3[0]
        elif action[10] == 0:
            trans_rate_BS0 = trans_rate_list_user5[0]
    elif len(trans_BS0_index) == 0:#没有用户连接
        trans_rate_BS0 = 0

    #BS1
    trans_BS1 = np.where(last_five_column == 1)[0]
    trans_BS1_index = trans_BS1 + (len(action) - len(last_five_columns))  # [6,7,8] 哪个user连接的
    if len(trans_BS1_index) == 3:#user1,2,5(连接了3个user)
        trans_rate_BS1 = trans_rate_list_user1[1]+trans_rate_list_user2[1] + trans_rate_list_user5[1]
    elif len(trans_BS1_index) == 2:#user12/15/25(连接了2个user)
        if action[6] == 1 and action[7] == 1:#user1,2
            trans_rate_BS1 = trans_rate_list_user1[1] + trans_rate_list_user2[1]
        elif action[6] == 1 and action[10] == 1:#user1,5
            trans_rate_BS1 = trans_rate_list_user1[1] + trans_rate_list_user5[1]
        elif action[7] == 1 and action[10] == 1:#user2,5
            trans_rate_BS1 = trans_rate_list_user2[1] + trans_rate_list_user5[1]
    elif len(trans_BS1_index) == 1:#只连接一个user1/2/5
        if action[6] == 1:
            trans_rate_BS1 = trans_rate_list_user1[1]
        elif action[7] == 1:
            trans_rate_BS1 = trans_rate_list_user2[1]
        elif action[10] == 1:
            trans_rate_BS1 = trans_rate_list_user5[1]
    elif len(trans_BS1_index) == 0:#一个user也没有连接
        trans_rate_BS1 = 0

    #BS2
    trans_BS2 = np.where(last_five_column == 2)[0]
    trans_BS2_index = trans_BS2 + (len(action) - len(last_five_columns))  # [6,7,8] 哪个user连接的
    if len(trans_BS2_index) == 3: #连接三个user3，4，5
        trans_rate_BS2 = trans_rate_list_user3[2] + trans_rate_list_user4[2] + trans_rate_list_user5[2]
    elif len(trans_BS2_index) == 2:#只连接2个user，34/35/45
        if action[8] == 2 and action[9] == 2:#user3,4
            trans_rate_BS2 = trans_rate_list_user3[2] + trans_rate_list_user4[2]
        elif action[8] == 2 and action[10] == 2:#user3,5
            trans_rate_BS2 = trans_rate_list_user3[2] + trans_rate_list_user5[2]
        elif action[9] == 2 and action[10] == 2:#user4,5
            trans_rate_BS2 = trans_rate_list_user4[2] + trans_rate_list_user5[2]
    elif len(trans_BS2_index) ==1:#只连接1个user，3/4/5
        if action[8] == 2:#user3
            trans_rate_BS2 = trans_rate_list_user3[2]
        elif action[9] == 2:#user4
            trans_rate_BS2 = trans_rate_list_user4[2]
        elif action[10] == 2:#user5
            trans_rate_BS2 = trans_rate_list_user5[2]
    elif len(trans_BS2_index) == 0:#没有用user连接
        trans_rate_BS2 = 0

    return trans_rate_BS0,trans_rate_BS1,trans_rate_BS2
#return trans_rate_BS0,trans_rate_BS1,trans_rate_BS2

#首先将到达的数据放到一个matrix中，因为还要计算delay，matrix具有3列，分别为：1.到达数据 2.数据到达时间 3.数据处理完成的时间
def arrival_data_matrix(time,user1_data,user2_data,user3_data,user4_data,user5_data,action,arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2):
    #首先根据action判断当前时刻哪个BS与哪个user连接了，action的第6-11列记录了user1，user2，user3，user4，user5与基站的连接
    #因为action的后5列记录了user与BS的连接情况，因此先取action的后5列
    last_five_column = action[-5:]
    #BS0
    BS0_connect = np.where(last_five_column == 0)[0]
    BS0_connect_column_index = BS0_connect + (len(action) - len(last_five_columns))#[6,7,8] 哪个user连接的

    if len(BS0_connect_column_index) == 3:#连接user2，3，5：
        arrival_data_matrix0[time][0] = user2_data[time] + user3_data[time] + user5_data[time]
        arrival_data_matrix0[time][1] = time

    elif len(BS0_connect_column_index) == 2:#连接user23/25/35
        if BS0_connect_column_index[0] == 7 and BS0_connect_column_index[1] == 8:#连接user2，3
            arrival_data_matrix0[time][0] = user2_data[time] + user3_data[time]
            arrival_data_matrix0[time][1] = time
        elif BS0_connect_column_index[0] == 8 and BS0_connect_column_index[1] == 10:#连接user3,5
            arrival_data_matrix0[time][0] = user3_data[time] + user5_data[time]
            arrival_data_matrix0[time][1] = time
        elif BS0_connect_column_index[0] == 7 and BS0_connect_column_index[1] == 10:#连接user2,5
            arrival_data_matrix0[time][0] = user2_data[time] + user5_data[time]
            arrival_data_matrix0[time][1] = time

    elif len(BS0_connect_column_index) == 1:#连接user2/3/5
        if BS0_connect_column_index[0] == 7:#连接user2
            arrival_data_matrix0[time][0] = user2_data[time]
            arrival_data_matrix0[time][1] = time
        elif BS0_connect_column_index[0] == 8:#连接user3
            arrival_data_matrix0[time][0] = user3_data[time]
            arrival_data_matrix0[time][1] = time
        elif BS0_connect_column_index[0] == 10:#连接user5
            arrival_data_matrix0[time][0] = user5_data[time]
            arrival_data_matrix0[time][1] = time

    elif len(BS0_connect_column_index) == 0:#一个user也没连接
        arrival_data_matrix0[time][0] = 0
        arrival_data_matrix0[time][1] = time

    #BS1
    BS1_connect = np.where(last_five_column == 1)[0]
    BS1_connect_column_index = BS1_connect + (len(action) - len(last_five_columns))

    if len(BS1_connect_column_index) == 3:#连接user1，2，5
        arrival_data_matrix1[time][0] = user1_data[time] + user1_data[time] + user5_data[time]
        arrival_data_matrix1[time][1] = time
    elif len(BS1_connect_column_index) == 2:#连接user12/15/25
        if BS1_connect_column_index[0] == 6 and BS1_connect_column_index[0] == 7:#连接user1，2
            arrival_data_matrix1[time][0] = user1_data[time] + user2_data[time]
            arrival_data_matrix1[time][1] = time
        elif BS1_connect_column_index[0] == 6 and BS1_connect_column_index[0] == 10:#连接user1，5
            arrival_data_matrix1[time][0] = user1_data[time] + user5_data[time]
            arrival_data_matrix1[time][1] = time
        elif BS1_connect_column_index[0] == 7 and BS1_connect_column_index[0] == 10:#连接user2，5
            arrival_data_matrix1[time][0] = user2_data[time] + user5_data[time]
            arrival_data_matrix1[time][1] = time

    elif len(BS1_connect_column_index) == 1: #连接user1
        if BS1_connect_column_index[0] == 6:#连接user1
            arrival_data_matrix1[time][0] = user1_data[time]
            arrival_data_matrix1[time][1] = time
        elif  BS1_connect_column_index[0] == 7:#连接user2
            arrival_data_matrix1[time][0] = user2_data[time]
            arrival_data_matrix1[time][1] = time
        elif BS1_connect_column_index[0] == 10:  # 连接user5
            arrival_data_matrix1[time][0] = user5_data[time]
            arrival_data_matrix1[time][1] = time

    elif len(BS1_connect_column_index) == 0:#没有连接user
        arrival_data_matrix1[time][0] = 0
        arrival_data_matrix1[time][1] = time

    #BS2
    BS2_connect = np.where(last_five_column == 2)[0]
    BS2_connect_column_index = BS2_connect + (len(action) - len(last_five_columns))

    if len(BS2_connect_column_index) == 3:#连接user3，4，5
        arrival_data_matrix2[time][0] = user3_data[time] + user4_data[time] + user5_data[time]
        arrival_data_matrix2[time][1] = time

    elif len(BS2_connect_column_index) == 2:#连接user34/35/45
        if BS2_connect_column_index[0] == 8 and BS2_connect_column_index[1] == 9:#连接user3，4
            arrival_data_matrix2[time][0] = user3_data[time] + user4_data[time]
            arrival_data_matrix2[time][1] = time
        elif BS2_connect_column_index[0] == 8 and BS2_connect_column_index[1] == 10:#连接user3，5
            arrival_data_matrix2[time][0] = user3_data[time] + user5_data[time]
            arrival_data_matrix2[time][1] = time
        elif BS2_connect_column_index[0] == 9 and BS2_connect_column_index[1] == 10:#连接user4，5
            arrival_data_matrix2[time][0] = user4_data[time] + user5_data[time]
            arrival_data_matrix2[time][1] = time

    elif len(BS2_connect_column_index) == 1:#连接user3，4
        if BS2_connect_column_index[0] == 8:#连接user3
            arrival_data_matrix2[time][0] = user3_data[time]
            arrival_data_matrix2[time][1] = time
        elif BS2_connect_column_index[0] == 9:#连接user4
            arrival_data_matrix2[time][0] = user4_data[time]
            arrival_data_matrix2[time][1] = time
        elif BS2_connect_column_index[0] == 10:  # 连接user5
            arrival_data_matrix2[time][0] = user5_data[time]
            arrival_data_matrix2[time][1] = time

    elif len(BS2_connect_column_index) == 0:#一个也没连接
        arrival_data_matrix2[time][0] = 0
        arrival_data_matrix2[time][1] = time
    return arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2
#return arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2

#定义waitpacketsize的大小
def wait_packetSize(time,arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2):
    #每个BS的 wait_packetSize为 = pending + 当前输入的，其中：pending = Vars.start_trans_no到（time-1）， Vars.start_trans_no表示还在处理的数
    wait_packetSize0 = sum(arrival_data_matrix0[Vars.start_trans_no:time,0])
    wait_packetSize1 = sum(arrival_data_matrix1[Vars.start_trans_no:time,0])
    wait_packetSize2 = sum(arrival_data_matrix2[Vars.start_trans_no:time,0])
    return wait_packetSize0, wait_packetSize1, wait_packetSize2
#return wait_packetSize0, wait_packetSize1, wait_packetSize2

#定义数据处理：数据减法的过程和功率消耗
def BS_power(time, action,BSmodel0,BSmodel1,BSmodel2, trans_rate0,transition_time_ratio0,trans_rate1,transition_time_ratio1,trans_rate2,transition_time_ratio2,wait_packetSize0, wait_packetSize1, wait_packetSize2,arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2,energy_sum,transition_time0,transition_time1,transition_time2):
    #按照基站，先一个基站一个基站的处理，action的第1-3列记录了BS0model，BS1model，BS2model（列index0，1，2）
    # BS0
    if transition_time0 != 0:  # 说明当前处于work model转换阶段
        if 0 < transition_time0 < 1:  # 说明当前处于模过渡阶段，且过渡时间transition_time_ratio0小于1ms
            transition_time_ratio0 = transition_time0
            transition_time0 = 0
        elif transition_time0 > 1:  # 当前处于模式过渡阶段且过渡时间大于1ms
            transition_time0 = transition_time0 - 1
    else:
        if BSmodel0 == 0: #activate  #这里不能根据state中的BSmodel来判断，要根据action中的model来判断，因为这是根据action转换到下一个stata的过程
            available_trans0 = trans_rate0 * (1 - transition_time_ratio0)
            if wait_packetSize0 >= available_trans0: #说明使用当前全部的1ms才能刚好处理或处理不完data
                #先判断zooming的大小
                if action[3] == 1: #当前处于red圆
                    #计算energy
                    power_coms0 = Vars.power_trans_full_red * (1-transition_time_ratio0)
                elif action[3] == 2: #当前处于black圆
                    power_coms0 = Vars.power_trans_full_black * (1 - transition_time_ratio0)
                elif action[3] == 3: #当前处于green圆
                    power_coms0 = Vars.power_trans_full_green  * (1 - transition_time_ratio0)
                energy_sum += power_coms0 #将当前时刻处理数据消耗的功耗加到总功耗中，用于计算reward
                trans_flag0 = True #用True标记一直传输
                tran_no0 = Vars.start_trans_no0#从arrival_data_matrix中的哪一行开始处理数据
                sum_trans0 = 0 #指当前1ms内已经处理完的数据
                while trans_flag0 == True and tran_no0 < len(arrival_data_matrix0):#还在处理数据切合法
                    if arrival_data_matrix0[trans_no][0]>0: #指当前arrival_data_matrix0行的数据还没处理完，如第1行100，处理完50，还剩50时
                        if sum_trans0 + arrival_data_matrix0[trans_no][0] > available_trans0:# sum_trans0指当前1ms内已经消耗的处理能力，arrival_data_matrix0[trans_no][0]指当前行的剩余数据
                            #如果当前1ms消耗的处理能力+matrix当前行的剩余数据>处理能力，说明当前的1ms处理不完arrival_data_matrix0[trans_no][0]的数据
                            arrival_data_matrix0[trans_no][0] = arrival_data_matrix0[trans_no][0] - (available_trans0 - sum_trans0)
                            trans_flag0 = False#当前1ms已经消耗完
                        else:#如果sum_trans0 + arrival_data_matrix0[trans_no][0] 小于 available_trans0，说明当前1ms内剩余的处理能力可以把当前的arrival_data_matrix0[trans_no][1]的数据处理完
                            sum_trans0 = sum_trans0 + arrival_data_matrix0[trans_no][0]
                            arrival_data_matrix0[trans_no][0] = 0 #当当前行的数据被处理完
                            Vars.start_trans_no0 = Vars.start_trans_no0 + 1#转入下一行处理数据
                            arrival_data_matrix0[trans_no][2] = time#记录处理完成数据的时刻，用于计算latency
                    #在else中没有trans_flag0设置为false（因为当前的1ms有能力处理完当前数据，而我们不确定当前1ms是否还可以继续处理数据，因此循环继续处理）
                    trans_no0 += 1#转入下一行处理数据
            elif  (wait_packetSize0 > 0) and (wait_packetSize0 < available_trans0): #说明当前的1ms完全可以处理完当前的数据，且处理完后因为没有立刻转入SM状态，因此在activate中有一段空闲时刻
                if action[3] == 1:
                    power_coms0 = (Vars.power_trans_full_red* (wait_packetSize0/available_trans0) + Vars.power_idle_red * (1-wait_packetSize0/trans_rate0))*(1-transition_time_ratio0)/1000  #两部分功率：传输过程中 + 空闲状态
                elif action[3] == 2:
                    power_coms0 = (Vars.power_trans_full_black * (wait_packetSize0 / available_trans0) + Vars.power_idle_black * (1 - wait_packetSize0 / trans_rate0)) * (1 - transition_time_ratio0) / 1000
                elif action[3] == 3:
                    power_coms0 = (Vars.power_trans_full_green * (wait_packetSize0 / available_trans0) + Vars.power_idle_green * (1 - wait_packetSize0 / trans_rate0)) * (1 - transition_time_ratio0) / 1000
                energy_sum += power_coms0 #将功率加进来
                trans_flag0 = True
                trans_no0 = Vars.start_trans_no0
                sum_trans0 = 0
                while trans_flag0 == True and tran_no0 < len(arrival_data_matrix0) and sum_trans0 < wait_packetSize0:#当前时刻已经有一部分处理的数据了，但是由于大的判断条件为wait_packetSize0 < available_trans0，所以这里要判断一下sum_trans0 < wait_packetSize0，说明当前1ms内还有一部分没有处理完的waitpacketsize
                    if arrival_data_matrix0[trans_no0][0] > 0:
                        if sum_trans0 + arrival_data_matrix0[trans_no0][0] > available_trans0:
                            arrival_data_matrix0[trans_no0][0] = arrival_data_matrix0[trans_no0][0] -(available_trans0-sum_trans0)
                            trans_flag0 = False
                        else:
                            sum_trans0 = sum_trans0 +  arrival_data_matrix0[trans_no][0]
                            arrival_data_matrix0[trans_no][0] = 0
                            Vars.start_trans_no0 = Vars.start_trans_no0 + 1
                            arrival_data_matrix0[trans_no][2] = time
                    trans_no0 += 1
            else:#没有waitpacketSize
                if action[3] == 1:
                    power_coms0 = Vars.power_idle_red * (1-transition_time_ratio0) #没有等待的包，则传输功率为空闲状态减去转换的时间
                elif action[3] == 2:
                    power_coms0 = Vars.power_idle_black * (1 - transition_time_ratio0)
                elif action[3] == 3:
                    power_coms0 = Vars.power_idle_green * (1 - transition_time_ratio0)
                energy_sum += power_coms0
        elif BSmodel0 == 1: #BS0处于SM1状态
            power_coms0 = Vars.power_SM1 * (1-transition_time_ratio0)  #除去转换的时间，剩余的就是睡眠时间了
            energy_sum += power_coms0  #将能耗加到总能耗中
        elif BSmodel0 == 2:  #SM2
            power_coms0 = Vars.power_SM2 * (1-transition_time_ratio0)
            energy_sum += power_coms0
        elif BSmodel0 == 3:  #SM3
            power_coms0 = Vars.power_SM2 * (1-transition_time_ratio0)
            energy_sum += power_coms0
        transition_time_ratio0 = 0

    # BS1
    if transition_time1 != 0:  # 说明当前处于work model转换阶段
        if 0 < transition_time1 < 1:  # 说明当前处于模过渡阶段，且过渡时间transition_time_ratio0小于1ms
            transition_time_ratio1 = transition_time1
            transition_time1 = 0
        elif transition_time1 > 1:  # 当前处于模式过渡阶段且过渡时间大于1ms
            transition_time1 = transition_time1 - 1
    else:
        if BSmodel1 == 0:
            available_trans1 = trans_rate1 * (1 - transition_time_ratio1)
            if wait_packetSize1 >= available_trans1:
                if action[4] == 1: #当前处于red圆
                    #计算energy
                    power_coms1 = Vars.power_trans_full_red * (1-transition_time_ratio1)
                elif action[4] == 2: #当前处于black圆
                    power_coms1 = Vars.power_trans_full_black * (1 - transition_time_ratio1)
                elif action[4] == 3: #当前处于green圆
                    power_coms1 = Vars.power_trans_full_green * (1 - transition_time_ratio1)
                # 计算energy
                energy_sum += power_coms1
                trans_flag1 = True
                tran_no1 = Vars.start_trans_no1
                sum_trans1 = 0
                while trans_flag1 == True and tran_no1 < len(arrival_data_matrix1):
                    if arrival_data_matrix1[trans_no][0] > 0:
                        if sum_trans1 + arrival_data_matrix1[trans_no][0] > available_trans1:
                            arrival_data_matrix1[trans_no][0] = arrival_data_matrix1[trans_no][0] - (available_trans1 - sum_trans1)
                            trans_flag1 = False
                        else:
                            sum_trans1 = sum_trans1 + arrival_data_matrix1[trans_no][0]
                            arrival_data_matrix1[trans_no][0] = 0
                            Vars.start_trans_no1 = Vars.start_trans_no1 + 1
                            arrival_data_matrix1[trans_no][2] = time
                    trans_no1 += 1
            elif (wait_packetSize1 > 0) and (wait_packetSize1 < available_trans1):
                if action[4] == 1:
                    power_coms1 = (Vars.power_trans_full_red * (wait_packetSize1 / available_trans1) + Vars.power_idle_red * (1 - wait_packetSize1 / trans_rate1)) * (1 - transition_time_ratio1) / 1000
                elif action[4] == 2:
                    power_coms1 = (Vars.power_trans_full_black * (wait_packetSize1 / available_trans1) + Vars.power_idle_black * (1 - wait_packetSize1 / trans_rate1)) * (1 - transition_time_ratio1) / 1000
                elif action[4] == 3:
                    power_coms1 = (Vars.power_trans_full_green * (wait_packetSize1 / available_trans1) + Vars.power_idle_green * (1 - wait_packetSize1 / trans_rate1)) * (1 - transition_time_ratio1) / 1000
                energy_sum += power_coms1
                trans_flag1 = True
                trans_no1 = Vars.start_trans_no1
                sum_trans1 = 0
                while trans_flag1 == True and tran_no1 < len(arrival_data_matrix1) and sum_trans1 < wait_packetSize1:
                    if arrival_data_matrix1[trans_no0][0] > 0:
                        if sum_trans1 + arrival_data_matrix1[trans_no0][0] > available_trans1:
                            arrival_data_matrix1[trans_no0][0] = arrival_data_matrix1[trans_no0][0] - (available_trans1 - sum_trans1)
                            trans_flag1 = False
                        else:
                            sum_trans1 = sum_trans1 + arrival_data_matrix1[trans_no][0]
                            arrival_data_matrix1[trans_no][0] = 0
                            Vars.start_trans_no1 = Vars.start_trans_no1 + 1
                            arrival_data_matrix1[trans_no][2] = time
                    trans_no1 += 1
            else:
                if action[4] == 1:
                    power_coms1 = Vars.power_idle_red * (1 - transition_time_ratio1)
                elif action[4] == 2:
                    power_coms1 = Vars.power_idle_black * (1 - transition_time_ratio1)
                elif action[4] == 3:
                    power_coms1 = Vars.power_idle_green * (1 - transition_time_ratio1)
                energy_sum += power_coms1
        elif BSmodel1 == 1:  # BS0处于SM1状态
            power_coms1 = Vars.power_SM1 * (1 - transition_time_ratio1)
            energy_sum += power_coms1
        elif BSmodel1 == 2:  # SM2
            power_coms1 = Vars.power_SM2 * (1 - transition_time_ratio1)
            energy_sum += power_coms1
        elif BSmodel1 == 3:  # SM3
            power_coms1 = Vars.power_SM2 * (1 - transition_time_ratio1)
            energy_sum += power_coms1
        transition_time_ratio1 = 0

    # BS2
    if transition_time2 != 0:  # 说明当前处于work model转换阶段
        if 0 < transition_time2 < 1:  # 说明当前处于模过渡阶段，且过渡时间transition_time_ratio0小于1ms
            transition_time_ratio2 = transition_time2
            transition_time2 = 0
        elif transition_time2 > 1:  # 当前处于模式过渡阶段且过渡时间大于1ms
            transition_time2 = transition_time2 - 1
    else:
        if BSmodel2 == 0:
            available_trans2 = trans_rate2 * (1 - transition_time_ratio2)
            if wait_packetSize2 >= available_trans2:
                if action[5] == 1: #当前处于red圆
                    #计算energy
                    power_coms2 = Vars.power_trans_full_red * (1-transition_time_ratio2)
                elif action[5] == 2: #当前处于black圆
                    power_coms2 = Vars.power_trans_full_black * (1 - transition_time_ratio2)
                elif action[5] == 3: #当前处于green圆
                    power_coms2 = Vars.power_trans_full_green * (1 - transition_time_ratio2)
                energy_sum += power_coms2
                trans_flag2 = True
                tran_no2 = Vars.start_trans_no2
                sum_trans2 = 0
                while trans_flag2 == True and tran_no2 < len(arrival_data_matrix2):
                    if arrival_data_matrix2[trans_no][0] > 0:
                        if sum_trans2 + arrival_data_matrix2[trans_no][0] > available_trans2:
                            arrival_data_matrix2[trans_no][0] = arrival_data_matrix2[trans_no][0] - (available_trans2 - sum_trans2)
                            trans_flag2 = False
                        else:
                            sum_trans2 = sum_trans2 + arrival_data_matrix2[trans_no][0]
                            arrival_data_matrix2[trans_no][0] = 0
                            Vars.start_trans_no2 = Vars.start_trans_no2 + 1
                            arrival_data_matrix2[trans_no][2] = time
                    trans_no2 += 1
            elif (wait_packetSize2 > 0) and (wait_packetSize2 < available_trans2):
                if action[5] == 1:
                    power_coms2 = (Vars.power_trans_full_red * (wait_packetSize2 / available_trans2) + Vars.power_idle_red * (1 - wait_packetSize2 / trans_rate2)) * (1 - transition_time_ratio2) / 1000
                elif action[5] == 2:
                    power_coms2 = (Vars.power_trans_full_black * (wait_packetSize2 / available_trans2) + Vars.power_idle_black * (1 - wait_packetSize2 / trans_rate2)) * (1 - transition_time_ratio2) / 1000
                elif action[5] == 3:
                    power_coms2 = (Vars.power_trans_full_green * (wait_packetSize2 / available_trans2) + Vars.power_idle_green * (1 - wait_packetSize2 / trans_rate2)) * (1 - transition_time_ratio2) / 1000
                energy_sum += power_coms2
                trans_flag2 = True
                trans_no2 = Vars.start_trans_no2
                sum_trans2 = 0
                while trans_flag2 == True and tran_no2 < len(arrival_data_matrix2) and sum_trans2 < wait_packetSize2:
                    if arrival_data_matrix2[trans_no0][0] > 0:
                        if sum_trans2 + arrival_data_matrix2[trans_no0][0] > available_trans2:
                            arrival_data_matrix2[trans_no0][0] = arrival_data_matrix2[trans_no0][0] - (available_trans2 - sum_trans2)
                            trans_flag2 = False
                        else:
                            sum_trans2 = sum_trans2 + arrival_data_matrix2[trans_no][0]
                            arrival_data_matrix2[trans_no][0] = 0
                            Vars.start_trans_no2 = Vars.start_trans_no2 + 1
                            arrival_data_matrix2[trans_no][2] = time
                    trans_no2 += 1
            else:
                if action[5] == 1:
                    power_coms2 = Vars.power_idle_red * (1 - transition_time_ratio2)
                elif action[5] == 2:
                    power_coms2 = Vars.power_idle_black * (1 - transition_time_ratio2)
                elif action[5] == 3:
                    power_coms2 = Vars.power_idle_green * (1 - transition_time_ratio2)
                energy_sum += power_coms2
        elif BSmodel2 == 1:  # BS0处于SM1状态
            power_coms2 = Vars.power_SM1 * (1 - transition_time_ratio2)
            energy_sum += power_coms2
        elif BSmodel2 == 2:  # SM2
            power_coms2 = Vars.power_SM2 * (1 - transition_time_ratio2)
            energy_sum += power_coms2
        elif BSmodel2 == 3:  # SM3
            power_coms2 = Vars.power_SM2 * (1 - transition_time_ratio2)
            energy_sum += power_coms2
        transition_time_ratio2 = 0
    return transition_time0, transition_time1, transition_time2, transition_time_ratio0,transition_time_ratio1,transition_time_ratio2, energy_sum,arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2
#return  transition_time_ratio0,transition_time_ratio1,transition_time_ratio2, energy_sum,arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2

#定义基站间模式转换的函数activate-SM1-SM2-SM3
def transition(BSmodel0,BSmodel1,BSmodel2,action,energy_sum,transition_time_ratio0,transition_time_ratio1,transition_time_ratio2,transition_time0,transition_time1,transition_time2,mode_time0,mode_time1,mode_time2):#注意：这里的BSmodel0,BSmodel1,BSmodel2是指的当前时刻的state中的元素
    #首先判断当前时刻BS work model是什么，再根据action将BS work model转换为新的work model
    #BS0
    if transition_time0 != 0:
        pass
    else:
        if BSmodel0 == 0:#Avtivate
            if action[0] == 1: #activate-SM1
                BSmodel0 = 1
                transition_time_ratio0 = Vars.delay_Active_SM1 #转换时间为0.0355
                if action[3] == 1:
                    power_coms_trans0 = Vars.power_idle_red * transition_time_ratio0#计算转换时的功率
                elif action[3] ==2:
                    power_coms_trans0 = Vars.power_idle_black * transition_time_ratio0
                elif action[3] ==3:
                    power_coms_trans0 = Vars.power_idle_green * transition_time_ratio0
                energy_sum += power_coms_trans0
                transition_time0 = 0 #转换完后，将转换时间重置为0
                mode_time0 = 0#指model的持续时间
            elif action[0] == 2: #activate-SM2,则要先经过SM1
                BSmodel0 = 2
                transition_time_ratio0 = Vars.delay_Active_SM1 + Vars.delay_SM1_SM2 #0.0355+0.5
                if action[3] == 1:
                    power_coms_trans0 = Vars.power_idle_red * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 #activate功率*activate—SM1的延迟时间 + SM1的时间*SM1_SM2的延迟  注：既然是延迟，就是如SM1-SM2，就是在SM1停留延迟时间后，再转换到SM2
                elif action[3] == 2:
                    power_coms_trans0 = Vars.power_idle_black * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2
                elif action[3] == 3:
                    power_coms_trans0 = Vars.power_idle_green * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2
                energy_sum += power_coms_trans0
                SM_Hold_Time0 = Vars.hold_time_SM2
                transition_time0 = SM_Hold_Time0-(1-transition_time_ratio0)
                energy_sum += Vars.power_SM2 * Vars.hold_time_SM2 #SM2的hole功率
                mode_time0 = 0
            elif action[0] == 3:  # active-SM3
                BSmodel0 = 3
                if action[3] == 1:
                    power_coms_transition0 = Vars.power_idle_red * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                elif action[3] == 2:
                    power_coms_transition0 = Vars.power_idle_black * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                elif action[3] == 3:
                    power_coms_transition0 = Vars.power_idle_green * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                energy_sum += power_coms_transition0
                transition_time0 = Vars.delay_Active_SM1 + Vars.delay_SM1_SM2 + Vars.delay_SM2_SM3
                mode_time0 = 0
            else:#继续保持activate
                mode_time0 += 1
                transition_time_ratio0 = 0
                transition_time0 = 0

        elif BSmodel0 == 1:#SM1
            if action[0] == 0: #SM1-Activate
                BSmodel0 = 0
                transition_time_ratio0 = Vars.delay_Active_SM1
                power_coms_transition0 = Vars.power_SM1 * transition_time_ratio0
                energy_sum += power_coms_transition0
                transition_time0 = 0
                mode_time0 = 0
            elif action[0] == 2: #SM1-SM2
                BSmodel0 = 2
                transition_time_ratio0 = Vars.delay_SM1_SM2
                power_coms_transition0 = Vars.power_SM1 * transition_time_ratio0
                energy_sum += power_coms_transition0
                SM_Hold_Time0 = Vars.hold_time_SM2
                transition_time0 = SM_Hold_Time0 - (1 - transition_time_ratio0)#?
                energy_sum += Vars.power_SM2 * Vars.hold_time_SM2
                mode_time0 = 0
            elif action[0] == 3: #SM1-SM3
                BSmodel0 = 3
                power_coms_transition0 = Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                energy_sum += power_coms_transition0
                transition_timee0 = Vars.delay_SM1_SM2 + Vars.delay_SM2_SM3
                mode_time0 = 0
            else:
                mode_time0 += 1  #持续当前的 BS mode = 1
                transition_time_ratio0 = 0
                transition_time0 = 0
        elif BSmodel0 == 2: #SM2
            if action[0] == 0: #SM2-Activate
                BSmodel0 =0
                transition_time_ratio0 = Vars.delay_SM1_SM2 + Vars.hold_time_SM1
                power_coms_transition0 = Vars.power_SM2 * Vars.delay_SM1_SM2 + Vars.power_SM1 * Vars.hold_time_SM1
                energy_sum += power_coms_transition0
                if transition_time0 > Vars.delay_SM1_SM2:
                    energy_sum = energy_sum - Vars.power_SM2 * Vars.delay_SM1_SM2
                    transition_time_ratio0 = transition_time0 + Vars.hold_time_SM1
                    if transition_time_ratio0 > 1:
                        transition_time0 = transition_time_ratio0
                else:
                    energy_sum = energy_sum - Vars.power_SM2 * transition_time0
                mode_time0 = 0
            elif action[0] == 1:  # SM2-SM1
                BSmodel0 = 1
                transition_time_ratio0 = Vars.delay_SM1_SM2
                power_coms_transition0 = Vars.power_SM2 * transition_time_ratio0
                energy_sum += power_coms_transition0
                if transition_time0 > Vars.delay_SM1_SM2:
                    energy_sum = energy_sum - Vars.power_SM2 * Vars.delay_SM1_SM2
                    transition_time_ratio0 = transition_time0
                    if transition_time_ratio0 > 1:
                        transition_time0 = transition_time_ratio0
                else:
                    energy_sum = energy_sum - Vars.power_SM2 * transition_time0
                mode_time0 = 0

            elif action[0] == 3:  # SM2-SM3
                BSmodel0 = 3
                power_coms_transition0 = Vars.power_SM2 * Vars.delay_SM2_SM3
                energy_sum += power_coms_transition0
                energy_sum = energy_sum - Vars.power_SM2 * transition_time0
                transition_time0 = transition_time0 + Vars.delay_SM2_SM3
                mode_time0 = 0
            else:
                mode_time0 += 1
                transition_time_ratio0 = 0
                transition_time0 = 0

        elif BSmodel0 == 3: #SM3 mode
            if action[0] == 0:  # SM3-active
                BSmodel0 = 0
                power_coms_transition0 = Vars.power_SM3 * Vars.delay_SM2_SM3 + Vars.power_SM2 * Vars.hold_time_SM2 + Vars.power_SM1 * Vars.hold_time_SM1
                energy_sum += power_coms_transition0
                transition_time0 = Vars.delay_SM2_SM3 + Vars.hold_time_SM2 + Vars.hold_time_SM1
                mode_time0 = 0
            elif action[0] == 1:  # SM3-SM1
                BSmodel0 = 1
                power_coms_transition0 = Vars.power_SM3 * Vars.delay_SM2_SM3 + Vars.power_SM2 * Vars.hold_time_SM2
                energy_sum += power_coms_transition0
                transition_time0 = Vars.delay_SM2_SM3 + Vars.hold_time_SM2
                mode_time0 = 0
            elif action[0] == 2:  # SM3-SM2
                BSmodel0 = 2
                power_coms_transition0 = Vars.power_SM3 * Vars.delay_SM2_SM3
                energy_sum += power_coms_transition0
                transition_time0 = Vars.delay_SM2_SM3 + Vars.hold_time_SM2
                energy_sum += Vars.power_SM2 * Vars.hold_time_SM2
                mode_time0 = 0
            else:
                mode_time0 += 1
                transition_time_ratio0 = 0
                transition_time0 = 0

    #BS1
    if transition_time1 != 0:
        pass
    else:
        if BSmodel1 == 0:#Avtivate
            if action[1] == 1: #activate-SM1
                BSmodel1 = 1
                transition_time_ratio1 = Vars.delay_Active_SM1 #转换时间为0.0355
                if action[4] == 1:
                    power_coms_trans1 = Vars.power_idle_red * transition_time_ratio1#计算转换时的功率
                elif action[4] == 2:
                    power_coms_trans1 = Vars.power_idle_black * transition_time_ratio1
                elif action[4] == 3:
                    power_coms_trans1 = Vars.power_idle_green * transition_time_ratio1
                energy_sum += power_coms_trans1
                transition_time1 = 0 #转换完后，将转换时间重置为0
                mode_time1 = 0#指model的持续时间
            elif action[1] == 2: #activate-SM2,则要先经过SM1
                BSmodel1 = 2
                transition_time_ratio1 = Vars.delay_Active_SM1 + Vars.delay_SM1_SM2 #0.0355+0.5
                if action[4] == 1:
                    power_coms_trans1 = Vars.power_idle_red * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 #activate功率*activate—SM1的延迟时间 + SM1的时间*SM1_SM2的延迟  注：既然是延迟，就是如SM1-SM2，就是在SM1停留延迟时间后，再转换到SM2
                elif action[4] == 2:
                    power_coms_trans1 = Vars.power_idle_black * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2
                if action[4] == 3:
                    power_coms_trans1 = Vars.power_idle_green * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2
                energy_sum += power_coms_trans1
                SM_Hold_Time1 = Vars.hold_time_SM2
                transition_time1 = SM_Hold_Time1 - (1-transition_time_ratio1)
                energy_sum += Vars.power_SM2 * Vars.hold_time_SM2 #SM2的hole功率
                mode_time1 = 0
            elif action[1] == 3:  # active-SM3
                BSmodel1 = 3
                if action[4] == 1:
                    power_coms_transition1 = Vars.power_idle_red * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                elif action[4] == 2:
                    power_coms_transition1 = Vars.power_idle_black * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                elif action[4] == 3:
                    power_coms_transition1 = Vars.power_idle_green * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                energy_sum += power_coms_transition1
                transition_time1 = Vars.delay_Active_SM1 + Vars.delay_SM1_SM2 + Vars.delay_SM2_SM3
                mode_time1 = 0
            else:#继续保持activate
                mode_time1 += 1
                transition_time_ratio1 = 0
                transition_time1 = 0

        elif BSmodel1 == 1:#SM1
            if action[1] == 0: #SM1-Activate
                BSmodel1 = 0
                transition_time_ratio1 = Vars.delay_Active_SM1
                power_coms_transition1 = Vars.power_SM1 * transition_time_ratio1
                energy_sum += power_coms_transition1
                transition_time1 = 0
                mode_time1 = 0
            elif action[1] == 2: #SM1-SM2
                BSmodel1 = 2
                transition_time_ratio1 = Vars.delay_SM1_SM2
                power_coms_transition1 = Vars.power_SM1 * transition_time_ratio1
                energy_sum += power_coms_transition1
                SM_Hold_Time1 = Vars.hold_time_SM2
                transition_time1 = SM_Hold_Time1- (1 - transition_time_ratio1)#?
                energy_sum += Vars.power_SM2 * Vars.hold_time_SM2
                mode_time1 = 0
            elif action[1] == 3: #SM1-SM3
                BSmodel1 = 3
                power_coms_transition1 = Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                energy_sum += power_coms_transition1
                transition_timee1 = Vars.delay_SM1_SM2 + Vars.delay_SM2_SM3
                mode_time1 = 0
            else:
                mode_time1 += 1  #持续当前的 BS mode = 1
                transition_time_ratio1 = 0
                transition_time1 = 0
        elif BSmodel1 == 2: #SM2
            if action[1] == 0: #SM2-Activate
                BSmodel1 =0
                transition_time_ratio1 = Vars.delay_SM1_SM2 + Vars.hold_time_SM1
                power_coms_transition1 = Vars.power_SM2 * Vars.delay_SM1_SM2 + Vars.power_SM1 * Vars.hold_time_SM1
                energy_sum += power_coms_transition1
                if transition_time1 > Vars.delay_SM1_SM2:
                    energy_sum = energy_sum - Vars.power_SM2 * Vars.delay_SM1_SM2
                    transition_time_ratio1 = transition_time1 + Vars.hold_time_SM1
                    if transition_time_ratio1 > 1:
                        transition_time1 = transition_time_ratio1
                else:
                    energy_sum = energy_sum - Vars.power_SM2 * transition_time1
                mode_time1 = 0
            elif action[1] == 1:  # SM2-SM1
                BSmodel1 = 1
                transition_time_ratio1 = Vars.delay_SM1_SM2
                power_coms_transition1 = Vars.power_SM2 * transition_time_ratio1
                energy_sum += power_coms_transition1
                if transition_time1 > Vars.delay_SM1_SM2:
                    energy_sum = energy_sum - Vars.power_SM2 * Vars.delay_SM1_SM2
                    transition_time_ratio1 = transition_time1
                    if transition_time_ratio1 > 1:
                        transition_time1 = transition_time_ratio1
                else:
                    energy_sum = energy_sum - Vars.power_SM2 * transition_time1
                mode_time1 = 0

            elif action[1] == 3:  # SM2-SM3
                BSmodel1 = 3
                power_coms_transition1 = Vars.power_SM2 * Vars.delay_SM2_SM3
                energy_sum += power_coms_transition1
                energy_sum = energy_sum - Vars.power_SM2 * transition_time1
                transition_time1 = transition_time1 + Vars.delay_SM2_SM3
                mode_time1 = 0
            else:
                mode_time1 += 1
                transition_time_ratio1 = 0
                transition_time1 = 0

        elif BSmodel1 == 3: #SM3 mode
            if action[1] == 0:  # SM3-active
                BSmodel1 = 0
                power_coms_transition1 = Vars.power_SM3 * Vars.delay_SM2_SM3 + Vars.power_SM2 * Vars.hold_time_SM2 + Vars.power_SM1 * Vars.hold_time_SM1
                energy_sum += power_coms_transition1
                transition_time1 = Vars.delay_SM2_SM3 + Vars.hold_time_SM2 + Vars.hold_time_SM1
                mode_time1 = 0
            elif action[1] == 1:  # SM3-SM1
                BSmodel1 = 1
                power_coms_transition1 = Vars.power_SM3 * Vars.delay_SM2_SM3 + Vars.power_SM2 * Vars.hold_time_SM2
                energy_sum += power_coms_transition1
                transition_time1= Vars.delay_SM2_SM3 + Vars.hold_time_SM2
                mode_time1 = 0
            elif action[1] == 2:  # SM3-SM2
                BSmodel1 = 2
                power_coms_transition1 = Vars.power_SM3 * Vars.delay_SM2_SM3
                energy_sum += power_coms_transition1
                transition_time1 = Vars.delay_SM2_SM3 + Vars.hold_time_SM2
                energy_sum += Vars.power_SM2 * Vars.hold_time_SM2
                mode_time1 = 0
            else:
                mode_time1 += 1
                transition_time_ratio1 = 0
                transition_time1 = 0

    #BS2
    if transition_time2 != 0:
        pass
    else:
        if BSmodel2 == 0:#Avtivate
            if action[2] == 1: #activate-SM1
                BSmodel2 = 1
                transition_time_ratio2 = Vars.delay_Active_SM1 #转换时间为0.0355
                if action[5] == 1:
                    power_coms_trans2 = Vars.power_idle_red * transition_time_ratio2#计算转换时的功率
                elif action[5] == 2:
                    power_coms_trans2 = Vars.power_idle_balck * transition_time_ratio2
                elif action[5] == 3:
                    power_coms_trans2 = Vars.power_idle_green * transition_time_ratio2
                energy_sum += power_coms_trans2
                transition_time2 = 0 #转换完后，将转换时间重置为0
                mode_time2 = 0#指model的持续时间
            elif action[2] == 2: #activate-SM2,则要先经过SM1
                BSmodel2 = 2
                transition_time_ratio2 = Vars.delay_Active_SM1 + Vars.delay_SM1_SM2 #0.0355+0.5
                if action[5] == 1:
                    power_coms_trans2 = Vars.power_idle_red * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 #activate功率*activate—SM1的延迟时间 + SM1的时间*SM1_SM2的延迟  注：既然是延迟，就是如SM1-SM2，就是在SM1停留延迟时间后，再转换到SM2
                elif action[5] == 2:
                    power_coms_trans2 = Vars.power_idle_black * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2
                elif action[5] == 3:
                    power_coms_trans2 = Vars.power_idle_green * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2
                energy_sum += power_coms_trans2
                SM_Hold_Time2 = Vars.hold_time_SM2
                transition_time2 = SM_Hold_Time2 - (1-transition_time_ratio2)
                energy_sum += Vars.power_SM2 * Vars.hold_time_SM2 #SM2的hole功率
                mode_time2 = 0
            elif action[2] == 3:  # active-SM3
                BSmodel2 = 3
                if action[5] == 1:
                    power_coms_transition2 = Vars.power_idle_red * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                elif action[5] == 2:
                    power_coms_transition2 = Vars.power_idle_black * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                elif action[5] == 3:
                    power_coms_transition2 = Vars.power_idle_green * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                energy_sum += power_coms_transition2
                transition_time2 = Vars.delay_Active_SM1 + Vars.delay_SM1_SM2 + Vars.delay_SM2_SM3
                mode_time2 = 0
            else:#继续保持activate
                mode_time2 += 1
                transition_time_ratio2 = 0
                transition_time2 = 0

        elif BSmodel2 == 1:#SM1
            if action[2] == 0: #SM1-Activate
                BSmodel2 = 0
                transition_time_ratio2 = Vars.delay_Active_SM1
                power_coms_transition2 = Vars.power_SM1 * transition_time_ratio2
                energy_sum += power_coms_transition2
                transition_time2 = 0
                mode_time2 = 0
            elif action[2] == 2: #SM1-SM2
                BSmodel2 = 2
                transition_time_ratio2 = Vars.delay_SM1_SM2
                power_coms_transition2 = Vars.power_SM1 * transition_time_ratio2
                energy_sum += power_coms_transition2
                SM_Hold_Time2 = Vars.hold_time_SM2
                transition_time2 = SM_Hold_Time2- (1 - transition_time_ratio2)#?
                energy_sum += Vars.power_SM2 * Vars.hold_time_SM2
                mode_time2 = 0
            elif action[2] == 3: #SM1-SM3
                BSmodel2 = 3
                power_coms_transition2 = Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                energy_sum += power_coms_transition2
                transition_timee2 = Vars.delay_SM1_SM2 + Vars.delay_SM2_SM3
                mode_time2 = 0
            else:
                mode_time2 += 1  #持续当前的 BS mode = 1
                transition_time_ratio2 = 0
                transition_time2 = 0
        elif BSmodel2 == 2: #SM2
            if action[2] == 0: #SM2-Activate
                BSmodel2 =0
                transition_time_ratio2 = Vars.delay_SM1_SM2 + Vars.hold_time_SM1
                power_coms_transition2 = Vars.power_SM2 * Vars.delay_SM1_SM2 + Vars.power_SM1 * Vars.hold_time_SM1
                energy_sum += power_coms_transition2
                if transition_time2 > Vars.delay_SM1_SM2:
                    energy_sum = energy_sum - Vars.power_SM2 * Vars.delay_SM1_SM2
                    transition_time_ratio2 = transition_time2 + Vars.hold_time_SM1
                    if transition_time_ratio2 > 1:
                        transition_time2 = transition_time_ratio2
                else:
                    energy_sum = energy_sum - Vars.power_SM2 * transition_time2
                mode_time2 = 0
            elif action[2] == 1:  # SM2-SM1
                BSmodel1 = 2
                transition_time_ratio2 = Vars.delay_SM1_SM2
                power_coms_transition2 = Vars.power_SM2 * transition_time_ratio2
                energy_sum += power_coms_transition2
                if transition_time2 > Vars.delay_SM1_SM2:
                    energy_sum = energy_sum - Vars.power_SM2 * Vars.delay_SM1_SM2
                    transition_time_ratio2 = transition_time2
                    if transition_time_ratio2 > 1:
                        transition_time2 = transition_time_ratio2
                else:
                    energy_sum = energy_sum - Vars.power_SM2 * transition_time2
                mode_time2 = 0

            elif action[2] == 3:  # SM2-SM3
                BSmodel2 = 3
                power_coms_transition2 = Vars.power_SM2 * Vars.delay_SM2_SM3
                energy_sum += power_coms_transition2
                energy_sum = energy_sum - Vars.power_SM2 * transition_time2
                transition_time2 = transition_time2 + Vars.delay_SM2_SM3
                mode_time2 = 0
            else:
                mode_time2 += 1
                transition_time_ratio2 = 0
                transition_time2 = 0

        elif BSmodel2 == 3: #SM3 mode
            if action[2] == 0:  # SM3-active
                BSmodel2 = 0
                power_coms_transition2 = Vars.power_SM3 * Vars.delay_SM2_SM3 + Vars.power_SM2 * Vars.hold_time_SM2 + Vars.power_SM1 * Vars.hold_time_SM1
                energy_sum += power_coms_transition2
                transition_time2 = Vars.delay_SM2_SM3 + Vars.hold_time_SM2 + Vars.hold_time_SM1
                mode_time2 = 0
            elif action[2] == 1:  # SM3-SM1
                BSmodel2 = 1
                power_coms_transition2 = Vars.power_SM3 * Vars.delay_SM2_SM3 + Vars.power_SM2 * Vars.hold_time_SM2
                energy_sum += power_coms_transition2
                transition_time2 = Vars.delay_SM2_SM3 + Vars.hold_time_SM2
                mode_time2 = 0
            elif action[2] == 2:  # SM3-SM2
                BSmodel2 = 2
                power_coms_transition2 = Vars.power_SM3 * Vars.delay_SM2_SM3
                energy_sum += power_coms_transition2
                transition_time2 = Vars.delay_SM2_SM3 + Vars.hold_time_SM2
                energy_sum += Vars.power_SM2 * Vars.hold_time_SM2
                mode_time2 = 0
            else:
                mode_time2 += 1
                transition_time_ratio2 = 0
                transition_time2 = 0
    return BSmodel0,BSmodel1,BSmodel2,energy_sum,transition_time_ratio0,transition_time_ratio1,transition_time_ratio2,transition_time0,transition_time1,transition_time2,mode_time0,mode_time1,mode_time2
#return BSmodel0,BSmodel1,BSmodel2,energy_sum,transition_time_ratio0,transition_time_ratio1,transition_time_ratio2,transition_time0,transition_time1,transition_time2,mode_time0,mode_time1,mode_time2




















