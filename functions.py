import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
    #如action为[0,0,3,3,1,0,-1,-1,-1,-1,-1]
    #在这种情况下，action为一个user也没连接，但是此时不耽误BS为activate状态，因为即使没有用户时，BS也可以是action状态的
    #但是trans_rate是根据用户来判断的，因此，需要先判断当前有没有user连接，有连接，才有trans_rate,没有连接，则就不存在trans_rate
    if action[6] != -1:#说明连接了，再算trans_rate
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
                elif action[3] ==0:#此时基站为休眠状态，SNR = 0
                    SNR = 0
            elif index == 1:#user1-BS1
                if action[4] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[4] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[4] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[4] == 0:
                    SNR = 0
            elif index == 2:#user1-BS2
                if action[5] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[5] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[5] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[5] == 0:
                    SNR = 0
            if SNR == 0:#不连接，此时则没有传输速率
                trans_rate = 0
            else:
                trans_rate = Vars.Bandwidth * np.log2(1 + SNR) / 1000 #in kbpms
            SNR_list_user1.append(SNR)
            trans_rate_list_user1.append(trans_rate)
    else:
        trans_rate_list_user1 = [0,0,0]

    #user2
    if action[7] != -1:
        for row,colu in enumerate(Vars.distance_matrix[1,:]):
            h = np.random.normal(loc=0, scale=1) / (colu**(Vars.alpha / 2))
            h_abs = np.abs(h)
            h_15.append(h_abs)
            h_squ =np.square(h_abs)
            if row == 0:#user2-BS0
                if action[3] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[3] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[3] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[3] == 0:
                    SNR = 0
            elif row == 1:#user2-BS1
                if action[4] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[4] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[4] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[4] == 0:
                    SNR = 0
            elif row == 2:#user2-BS2
                if action[5] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[5] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[5] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[5] == 0:
                    SNR = 0
            if SNR == 0:
                trans_rate = 0
            else:
                trans_rate = Vars.Bandwidth * np.log2(1 + SNR) / 1000 #in kbpms
            SNR_list_user2.append(SNR)
            trans_rate_list_user2.append(trans_rate)
    else:
        trans_rate_list_user2 = [0,0,0]
    #user3
    if action[8] != -1:
        for row,colu in enumerate(Vars.distance_matrix[2,:]):
            h = np.random.normal(loc=0, scale=1) / (colu**(Vars.alpha / 2))
            h_abs = np.abs(h)
            h_15.append(h_abs)
            h_squ =np.square(h_abs)
            if row == 0:#user3-BS0
                if action[3] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[3] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[3] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[3] == 0:
                    SNR = 0
            elif row == 1:#user3-BS1
                if action[4] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[4] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[4] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[4] == 0:
                    SNR= 0
            elif row == 2:#user3-BS2
                if action[5] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[5] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[5] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[5] == 0:
                    SNR = 0
            if SNR == 0:
                trans_rate = 0
            else:
                trans_rate = Vars.Bandwidth * np.log2(1 + SNR) / 1000 #in kbpms
            SNR_list_user3.append(SNR)
            trans_rate_list_user3.append(trans_rate)
    else:
        trans_rate_list_user3 = [0,0,0]
    #user4
    if action[9] != -1:
        for row,colu in enumerate(Vars.distance_matrix[3,:]):
            h = np.random.normal(loc=0, scale=1) / (colu**(Vars.alpha / 2))
            h_abs = np.abs(h)
            h_15.append(h_abs)
            h_squ =np.square(h_abs)
            if row == 0:#user4-BS0
                if action[3] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[3] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[3] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[3] == 0:
                    SNR = 0
            elif row == 1:#user2-BS1
                if action[4] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[4] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[4] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[4] == 0:
                    SNR = 0
            elif row == 2:#user-BS2
                if action[5] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[5] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[5] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[5] == 0:
                    SNR = 0
            if SNR ==0:
                trans_rate = 0
            else:
                trans_rate = Vars.Bandwidth * np.log2(1 + SNR) / 1000 #in kbpms
            SNR_list_user4.append(SNR)
            trans_rate_list_user4.append(trans_rate)
    else:
        trans_rate_list_user4 = [0,0,0]
    #user5
    if action[10] !=-1:
        for row,colu in enumerate(Vars.distance_matrix[4,:]):
            h = np.random.normal(loc=0, scale=1) / (colu**(Vars.alpha / 2))
            h_abs = np.abs(h)
            h_15.append(h_abs)
            h_squ =np.square(h_abs)
            if row == 0:#user4-BS0
                if action[3] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[3] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[3] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[3] == 0:
                    SNR = 0
            elif row == 1:#user2-BS1
                if action[4] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[4] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[4] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[4] == 0:
                    SNR = 0
            elif row == 2:#user-BS2
                if action[5] == 1:
                    SNR = Vars.power_trans_full_red * h_squ
                elif action[5] == 2:
                    SNR = Vars.power_trans_full_black * h_squ
                elif action[5] == 3:
                    SNR = Vars.power_trans_full_green * h_squ
                elif action[5] == 0:
                    SNR = 0
            if SNR == 0:
                trans_rate = 0
            else:
                trans_rate = Vars.Bandwidth * np.log2(1 + SNR) / 1000 #in kbpms
            SNR_list_user5.append(SNR)
            trans_rate_list_user5.append(trans_rate)
    else:
        trans_rate_list_user5 = [0,0,0]
    return h_15, trans_rate_list_user1,trans_rate_list_user2,trans_rate_list_user3,trans_rate_list_user4,trans_rate_list_user5
#return  h_15, trans_rate_list_user1,trans_rate_list_user2,trans_rate_list_user3,trans_rate_list_user4,trans_rate_list_user5

#根据action计算每个BS上的trans_rate
def trans_rate_3BS(action,trans_rate_list_user1,trans_rate_list_user2,trans_rate_list_user3,trans_rate_list_user4,trans_rate_list_user5):
    #获取action后5列的user-BS的连接情况
    last_five_columns = action[-5:]
    #BS0
    trans_BS0= np.where(last_five_columns == 0)[0]#判断那个用户与BS0连接
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
    trans_BS1 = np.where(last_five_columns == 1)[0]
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
    trans_BS2 = np.where(last_five_columns == 2)[0]
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
def arrival_data_matrix(user1_data,user2_data,user3_data,user4_data,user5_data,action,arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2):
    #首先根据action判断当前时刻哪个BS与哪个user连接了，action的第6-11列记录了user1，user2，user3，user4，user5与基站的连接
    #因为action的后5列记录了user与BS的连接情况，因此先取action的后5列
    #TODO  1.修改 last_five_column加上s   2.当当前用户没有数据却连接时，user_data的数据为空，无法计算，因此要加上判断用户数据是否为空的情况
    #TODO 3.要在take_action()加入对action的选择，1.waitpacketsize ！=0，去掉具有SM模式的action  2.userdata没有 去掉连接的action
    #TODO 这里还有一种情况就是，某个用户有到达数据，但是由于没有连接到基站，所有没有加入到arrival中，那这样的化，就直接用LSTM数据了，不用userdata的数据了只能
    # 列：arrivaldata_total/user1_data/user2_data/user3_data/user4_data/user5_data/arrival_time/process_time
    #          index0        index1     index2       index3      index4     index5    index6      index7
    #虽然没有数据到达，但是第6列的时间time是根据连接的action记录的，而不是usersdata，因此，虽然在当前时时间没有user到达，但是连接了，就有时间，而不是为-1
    last_five_columns = action[-5:]
    #BS0
    BS0_connect = np.where(last_five_columns == 0)[0]
    BS0_connect_column_index = BS0_connect + (len(action) - len(last_five_columns))#[6,7,8] 哪个user连接的

    if len(BS0_connect_column_index) == 3:#连接user2，3，5：
        arrival_data_matrix0[Vars.time][0] = user2_data[Vars.time] + user3_data[Vars.time] + user5_data[Vars.time]
        arrival_data_matrix0[Vars.time][2] = user2_data[Vars.time]
        arrival_data_matrix0[Vars.time][3] = user3_data[Vars.time]
        arrival_data_matrix0[Vars.time][5] = user5_data[Vars.time]

    elif len(BS0_connect_column_index) == 2:#连接user23/25/35
        if BS0_connect_column_index[0] == 7 and BS0_connect_column_index[1] == 8:#连接user2，3
            arrival_data_matrix0[Vars.time][0] = user2_data[Vars.time] + user3_data[Vars.time]
            arrival_data_matrix0[Vars.time][2] = user2_data[Vars.time]
            arrival_data_matrix0[Vars.time][3] = user3_data[Vars.time]
        elif BS0_connect_column_index[0] == 8 and BS0_connect_column_index[1] == 10:#连接user3,5
            arrival_data_matrix0[Vars.time][0] = user3_data[Vars.time] + user5_data[Vars.time]
            arrival_data_matrix0[Vars.time][3] = user3_data[Vars.time]
            arrival_data_matrix0[Vars.time][5] = user5_data[Vars.time]
        elif BS0_connect_column_index[0] == 7 and BS0_connect_column_index[1] == 10:#连接user2,5
            arrival_data_matrix0[Vars.time][0] = user2_data[Vars.time] + user5_data[Vars.time]
            arrival_data_matrix0[Vars.time][2] = user2_data[Vars.time]
            arrival_data_matrix0[Vars.time][5] = user5_data[Vars.time]

    elif len(BS0_connect_column_index) == 1:#连接user2/3/5
        if BS0_connect_column_index[0] == 7:#连接user2
            arrival_data_matrix0[Vars.time][0] = user2_data[Vars.time]
            arrival_data_matrix0[Vars.time][2] = user2_data[Vars.time]
        elif BS0_connect_column_index[0] == 8:#连接user3
            arrival_data_matrix0[Vars.time][0] = user3_data[Vars.time]
            arrival_data_matrix0[Vars.time][3] = user3_data[Vars.time]
        elif BS0_connect_column_index[0] == 10:#连接user5
            arrival_data_matrix0[Vars.time][0] = user5_data[Vars.time]
            arrival_data_matrix0[Vars.time][5] = user5_data[Vars.time]
    #这里是否可以加上，没有到达数据时，时间为0，用于标记呢
    elif len(BS0_connect_column_index) == 0:#一个user也没连接
        arrival_data_matrix0[Vars.time][0] = 0
        arrival_data_matrix0[Vars.time][6] = -1 #用于标记当前时刻没有数据到达
    if arrival_data_matrix0[Vars.time][0] != 0:
        arrival_data_matrix0[Vars.time][6] = Vars.time
    else:
        arrival_data_matrix0[Vars.time][6] = -1


    #BS1
    BS1_connect = np.where(last_five_columns == 1)[0]
    BS1_connect_column_index = BS1_connect + (len(action) - len(last_five_columns))

    if len(BS1_connect_column_index) == 3:#连接user1，2，5
        arrival_data_matrix1[Vars.time][0] = user1_data[Vars.time] + user1_data[Vars.time] + user5_data[Vars.time]
        arrival_data_matrix1[Vars.time][1] = user1_data[Vars.time]
        arrival_data_matrix1[Vars.time][2] = user2_data[Vars.time]
        arrival_data_matrix1[Vars.time][5] = user5_data[Vars.time]
    elif len(BS1_connect_column_index) == 2:#连接user12/15/25
        if BS1_connect_column_index[0] == 6 and BS1_connect_column_index[1] == 7:#连接user1，2
            arrival_data_matrix1[Vars.time][0] = user1_data[Vars.time] + user2_data[Vars.time]
            arrival_data_matrix1[Vars.time][1] = user1_data[Vars.time]
            arrival_data_matrix1[Vars.time][2] = user2_data[Vars.time]
        elif BS1_connect_column_index[0] == 6 and BS1_connect_column_index[1] == 10:#连接user1，5
            arrival_data_matrix1[Vars.time][0] = user1_data[Vars.time] + user5_data[Vars.time]
            arrival_data_matrix1[Vars.time][1] = user1_data[Vars.time]
            arrival_data_matrix1[Vars.time][5] = user5_data[Vars.time]
        elif BS1_connect_column_index[0] == 7 and BS1_connect_column_index[1] == 10:#连接user2，5
            arrival_data_matrix1[Vars.time][0] = user2_data[Vars.time] + user5_data[Vars.time]
            arrival_data_matrix1[Vars.time][2] = user2_data[Vars.time]
            arrival_data_matrix1[Vars.time][5] = user5_data[Vars.time]

    elif len(BS1_connect_column_index) == 1: #连接user1/2/5
        if BS1_connect_column_index[0] == 6:#连接user1
            arrival_data_matrix1[Vars.time][0] = user1_data[Vars.time]
            arrival_data_matrix1[Vars.time][1] = user1_data[Vars.time]
        elif  BS1_connect_column_index[0] == 7:#连接user2
            arrival_data_matrix1[Vars.time][0] = user2_data[Vars.time]
            arrival_data_matrix1[Vars.time][2] = user2_data[Vars.time]
        elif BS1_connect_column_index[0] == 10:  # 连接user5
            arrival_data_matrix1[Vars.time][0] = user5_data[Vars.time]
            arrival_data_matrix1[Vars.time][5] = user5_data[Vars.time]

    elif len(BS1_connect_column_index) == 0:#没有连接user
        arrival_data_matrix1[Vars.time][0] = 0
        arrival_data_matrix1[Vars.time][6] = -1 #用于标记当前时刻没有数据到达
    if arrival_data_matrix1[Vars.time][0] != 0:
        arrival_data_matrix1[Vars.time][6] = Vars.time
    else:
        arrival_data_matrix1[Vars.time][6] = -1

    #BS2
    BS2_connect = np.where(last_five_columns == 2)[0]
    BS2_connect_column_index = BS2_connect + (len(action) - len(last_five_columns))
    #TODO 除了长度为4的外，其他情况都少了user1
    if len(BS2_connect_column_index) == 4:  # 连接user1，3，4，5
        arrival_data_matrix2[Vars.time][0] = user1_data[Vars.time] + user3_data[Vars.time] + user4_data[Vars.time] + user5_data[Vars.time]
        arrival_data_matrix2[Vars.time][1] = user1_data[Vars.time]
        arrival_data_matrix2[Vars.time][3] = user3_data[Vars.time]
        arrival_data_matrix2[Vars.time][4] = user4_data[Vars.time]
        arrival_data_matrix2[Vars.time][5] = user5_data[Vars.time]
    if len(BS2_connect_column_index) == 3:  # 连接user 134/125/345
        if BS2_connect_column_index[0] == 6 and BS2_connect_column_index[1] == 8 and BS2_connect_column_index[2] == 9:  # 134
            arrival_data_matrix2[Vars.time][0] = user1_data[Vars.time] + user3_data[Vars.time] + user4_data[Vars.time]
            arrival_data_matrix2[Vars.time][1] = user1_data[Vars.time]
            arrival_data_matrix2[Vars.time][3] = user3_data[Vars.time]
            arrival_data_matrix2[Vars.time][4] = user4_data[Vars.time]
        elif BS2_connect_column_index[0] == 6 and BS2_connect_column_index[1] == 7 and BS2_connect_column_index[2] == 10:  # 125
            arrival_data_matrix2[Vars.time][0] = user1_data[Vars.time] + user2_data[Vars.time] + user5_data[Vars.time]
            arrival_data_matrix2[Vars.time][1] = user1_data[Vars.time]
            arrival_data_matrix2[Vars.time][2] = user2_data[Vars.time]
            arrival_data_matrix2[Vars.time][5] = user5_data[Vars.time]
        elif BS2_connect_column_index[0] == 8 and BS2_connect_column_index[1] == 9 and BS2_connect_column_index[2] == 10:  # 345
            arrival_data_matrix2[Vars.time][0] = user3_data[Vars.time] + user4_data[Vars.time] + user5_data[Vars.time]
            arrival_data_matrix2[Vars.time][3] = user3_data[Vars.time]
            arrival_data_matrix2[Vars.time][4] = user4_data[Vars.time]
            arrival_data_matrix2[Vars.time][5] = user5_data[Vars.time]

    elif len(BS2_connect_column_index) == 2:#连接user13/14/15/34/35/45
        if BS2_connect_column_index[0] == 6 and BS2_connect_column_index[1] == 8:#连接user13
            arrival_data_matrix2[Vars.time][0] = user1_data[Vars.time] + user3_data[Vars.time]
            arrival_data_matrix2[Vars.time][1] = user1_data[Vars.time]
            arrival_data_matrix2[Vars.time][3] = user3_data[Vars.time]
        elif BS2_connect_column_index[0] == 6 and BS2_connect_column_index[1] == 9:#连接user14
            arrival_data_matrix2[Vars.time][0] = user1_data[Vars.time] + user4_data[Vars.time]
            arrival_data_matrix2[Vars.time][1] = user1_data[Vars.time]
            arrival_data_matrix2[Vars.time][4] = user4_data[Vars.time]
        elif BS2_connect_column_index[0] == 6 and BS2_connect_column_index[1] == 10:#连接user15
            arrival_data_matrix2[Vars.time][0] = user1_data[Vars.time] + user5_data[Vars.time]
            arrival_data_matrix2[Vars.time][1] = user1_data[Vars.time]
            arrival_data_matrix2[Vars.time][5] = user5_data[Vars.time]
        elif BS2_connect_column_index[0] == 8 and BS2_connect_column_index[1] == 9:#连接user3，4
            arrival_data_matrix2[Vars.time][0] = user3_data[Vars.time] + user4_data[Vars.time]
            arrival_data_matrix2[Vars.time][3] = user3_data[Vars.time]
            arrival_data_matrix2[Vars.time][4] = user4_data[Vars.time]
        elif BS2_connect_column_index[0] == 8 and BS2_connect_column_index[1] == 10:#连接user3，5
            arrival_data_matrix2[Vars.time][0] = user3_data[Vars.time] + user5_data[Vars.time]
            arrival_data_matrix2[Vars.time][3] = user3_data[Vars.time]
            arrival_data_matrix2[Vars.time][5] = user5_data[Vars.time]
        elif BS2_connect_column_index[0] == 9 and BS2_connect_column_index[1] == 10:#连接user4，5
            arrival_data_matrix2[Vars.time][0] = user4_data[Vars.time] + user5_data[Vars.time]
            arrival_data_matrix2[Vars.time][4] = user4_data[Vars.time]
            arrival_data_matrix2[Vars.time][5] = user5_data[Vars.time]

    elif len(BS2_connect_column_index) == 1:#连接user1, 3，4, 5
        if BS2_connect_column_index[0] == 6:#连接user3
            arrival_data_matrix2[Vars.time][0] = user1_data[Vars.time]
            arrival_data_matrix2[Vars.time][1] = user1_data[Vars.time]
        elif BS2_connect_column_index[0] == 8:#连接user3
            arrival_data_matrix2[Vars.time][0] = user3_data[Vars.time]
            arrival_data_matrix2[Vars.time][3] = user3_data[Vars.time]
        elif BS2_connect_column_index[0] == 9:#连接user4
            arrival_data_matrix2[Vars.time][0] = user4_data[Vars.time]
            arrival_data_matrix2[Vars.time][4] = user4_data[Vars.time]
        elif BS2_connect_column_index[0] == 10:  # 连接user5
            arrival_data_matrix2[Vars.time][0] = user5_data[Vars.time]
            arrival_data_matrix2[Vars.time][5] = user5_data[Vars.time]


    elif len(BS2_connect_column_index) == 0:#一个也没连接
        arrival_data_matrix2[Vars.time][0] = 0
        arrival_data_matrix2[Vars.time][6] = -1 #用于标记当前时刻没有数据到达
    if  arrival_data_matrix2[Vars.time][0] !=0:
        arrival_data_matrix2[Vars.time][6] = Vars.time
    else:
        arrival_data_matrix2[Vars.time][6] = -1
    return arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2
#return arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2

#定义waitpacketsize的大小
def wait_packetSize(arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2,observation):
    #每个BS的 wait_packetSize为 = pending + 当前输入的，其中：pending = Vars.start_trans_no到（Vars.time-1）， Vars.start_trans_no表示还在处理的数
    #TODO 这里如果Vars.time == Vars.start_trans_no，说明没有选择任何元素，所以这里要加一个判断条件
    #TODO 注意：这里的waitpacketsize不为0，而BS_power中的数据为0的原因可能是这里的Vars.start_trans_no0_2跟他的不一样，他哪里发生了变化，因为有了+1，所以我在想，是不是可以在这里修改，因为在main函数中，是他调用在先
    #TODO 是不是可以认为waitpacketSize是从o开始到当前时间的综合
    wait_packetSize = observation['wait_packetSize']

    # for BS0-user2
    wait_packetSize[1] = sum(arrival_data_matrix0[0:Vars.time+1,2]) #计算从第0行到当前时间的arrival_data,当time=0是，也是[0:1],不冲突
    # for BS0-user3
    wait_packetSize[2] = sum( arrival_data_matrix0[0:Vars.time+1,3])
    # for BS0-user5
    wait_packetSize[4] =sum(arrival_data_matrix0[0:Vars.time+1,5])

    # for BS1-user1
    wait_packetSize[5] =sum(arrival_data_matrix1[0:Vars.time+1,1])
    # for BS0-user2
    wait_packetSize[6] = sum(arrival_data_matrix1[0:Vars.time+1,2])
    # for BS0-user5
    wait_packetSize[9] = sum(arrival_data_matrix1[0:Vars.time+1,5])

    # for BS2-user1
    wait_packetSize[10] =sum(arrival_data_matrix2[0:Vars.time+1,1])
    # for BS0-user3
    wait_packetSize[12] = sum(arrival_data_matrix2[0:Vars.time+1,3])
    # for BS0-user4
    wait_packetSize[13] = sum(arrival_data_matrix2[0:Vars.time+1,4])
    # for BS0-user5
    wait_packetSize[14] = sum(arrival_data_matrix2[0:Vars.time+1,5])

    return wait_packetSize
#return wait_packetSize0, wait_packetSize1, wait_packetSize2

#定义数据处理：数据减法的过程和功率消耗
def BS_power(action,observation,transition_time_ratio0,transition_time_ratio1,transition_time_ratio2,arrival_data_matrix0,arrival_data_matrix1,arrival_data_matrix2,energy_sum,transition_time0,transition_time1,transition_time2,trans_rate_list_user1, trans_rate_list_user2, trans_rate_list_user3, trans_rate_list_user4, trans_rate_list_user5):
    BSmodel0 = action[0]
    BSmodel1 = action[1]
    BSmodel2 = action[2]

    #TODO 这里等待数据包的大小应该为15个了
    waitpacketSize = observation['wait_packetSize']
    #BS0
    wait_packetSize0_2 = waitpacketSize[1]
    wait_packetSize0_3 = waitpacketSize[2]
    wait_packetSize0_5 = waitpacketSize[4]
    wait_packetSize0 = wait_packetSize0_2 +wait_packetSize0_3+wait_packetSize0_5
    #BS1
    wait_packetSize1_1 = waitpacketSize[5]
    wait_packetSize1_2 = waitpacketSize[6]
    wait_packetSize1_5 = waitpacketSize[9]
    wait_packetSize1 = wait_packetSize1_1 +wait_packetSize1_2+ wait_packetSize1_5
    #BS2
    wait_packetSize2_1 = waitpacketSize[10]
    wait_packetSize2_3 = waitpacketSize[12]
    wait_packetSize2_4 = waitpacketSize[13]
    wait_packetSize2_5 = waitpacketSize[14]
    wait_packetSize2 = wait_packetSize2_1+wait_packetSize2_3+wait_packetSize2_4+wait_packetSize2_5

    #按照基站，先一个基站一个基站的处理，action的第1-3列记录了BS0model，BS1model，BS2model（列index0，1，2）
    # BS0
    if transition_time0 != 0:  # 说明当前处于work model转换阶段
        if 0 < transition_time0 < 1:  # 说明当前处于模过渡阶段，且过渡时间transition_time_ratio0小于1ms
            transition_time_ratio0 = transition_time0
            transition_time0 = 0
        elif transition_time0 > 1:  # 当前处于模式过渡阶段且过渡时间大于1ms
            transition_time0 = transition_time0 - 1
    else:
        if BSmodel0 == 0: #activate  #这里的BSmodel是state中的model而不是action的model，因为现在是判断基站处于什么状态下
            #连接了user2/3/5
            available_trans0_2 = trans_rate_list_user2[0] * (1 - transition_time_ratio0)#user2
            available_trans0_3 = trans_rate_list_user3[0] * (1 - transition_time_ratio0)#user3
            available_trans0_5 = trans_rate_list_user5[0] * (1 - transition_time_ratio0)#user5

            if wait_packetSize0 > 0:
                # for user2
                if wait_packetSize0_2 >= available_trans0_2: #说明使用当前全部的1ms才能刚好处理或处理不完data
                    #先判断zooming的大小
                    if action[3] == 1: #当前处于red圆
                        #计算energy
                        power_coms0_2 = Vars.power_trans_full_red * (1-transition_time_ratio0)
                    elif action[3] == 2: #当前处于black圆
                        power_coms0_2 = Vars.power_trans_full_black * (1 - transition_time_ratio0)
                    elif action[3] == 3: #当前处于green圆
                        power_coms0_2 = Vars.power_trans_full_green  * (1 - transition_time_ratio0)
                    energy_sum += power_coms0_2 #将当前时刻处理数据消耗的功耗加到总功耗中，用于计算reward
                    trans_flag0_2 = True #用True标记一直传输
                    #先判断一下，当前是否有数据，没有数据Vars.start_tran_no就得一直+1，直到加到有数据为止
                    #tran_no0_2取在当前user的这一列第一个为有数的行index
                    first_data_index0_2 = np.where(arrival_data_matrix0[:,2] !=0)
                    if first_data_index0_2 and len(first_data_index0_2[0]) > 0:#当前不为空，证明有数据，则tran_no0_2的值取第一个有数的行index
                        tran_no0_2 = first_data_index0_2[0][0]
                    else:#否则，tran_no0_2 = 0
                        tran_no0_2 = 0
                        print('first_data_index0_2',first_data_index0_2)
                    Vars.start_trans_no0_2 = tran_no0_2
                    sum_trans0_2 = 0 #指当前1ms内已经处理完的数据
                    while trans_flag0_2 == True and tran_no0_2 < len(arrival_data_matrix0):#还在处理数据切合法
                        if arrival_data_matrix0[tran_no0_2][2]>0: #user2有等待数据#指当前arrival_data_matrix0行的数据还没处理完，如第1行100，处理完50，还剩50时
                            if sum_trans0_2 + arrival_data_matrix0[tran_no0_2][2] > available_trans0_2:# sum_trans0指当前1ms内已经消耗的处理能力，arrival_data_matrix0[trans_no][0]指当前行的剩余数据
                                #如果当前1ms消耗的处理能力+matrix当前行的剩余数据>处理能力，说明当前的1ms处理不完arrival_data_matrix0[trans_no][0]的数据
                                arrival_data_matrix0[tran_no0_2][2] = arrival_data_matrix0[tran_no0_2][2] - (available_trans0_2 - sum_trans0_2)
                                trans_flag0_2 = False#当前1ms已经消耗完,这一秒已经过完了，直接退出这1ms，进入下1ms，下一次大循环了
                            else:#如果sum_trans0 + arrival_data_matrix0[trans_no][0] 小于 available_trans0，说明当前1ms内剩余的处理能力可以把当前的arrival_data_matrix0[trans_no][1]的数据处理完
                                sum_trans0_2 = sum_trans0_2 + arrival_data_matrix0[tran_no0_2][2]
                                arrival_data_matrix0[tran_no0_2][2] = 0 #当当前行的数据被处理完
                                if arrival_data_matrix0[tran_no0_2][2] == 0 and arrival_data_matrix0[tran_no0_2][3] == 0 and arrival_data_matrix0[tran_no0_2][5] == 0:
                                    #当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix0[tran_no0_2][7] = Vars.time#记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index0_2 = np.where(arrival_data_matrix0[:, 2] != 0)
                                if first_data_index0_2 and len(first_data_index0_2[0]) > 0:
                                    tran_no0_2 = first_data_index0_2[0][0]
                                else:
                                    tran_no0_2 = 0
                                    print('first_data_index0_2', first_data_index0_2)
                                Vars.start_trans_no0_2 = tran_no0_2#转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix0[tran_no0_2][0] = arrival_data_matrix0[tran_no0_2][2] + \
                                                          arrival_data_matrix0[tran_no0_2][3] + \
                                                          arrival_data_matrix0[tran_no0_2][5]
                        #在else中没有trans_flag0设置为false（因为当前的1ms有能力处理完当前数据，而我们不确定当前1ms是否还可以继续处理数据，因此循环继续处理）
                elif  (wait_packetSize0_2 > 0) and (wait_packetSize0_2 < available_trans0_2): #说明当前的1ms完全可以处理完当前的数据，且处理完后因为没有立刻转入SM状态，因此在activate中有一段空闲时刻
                    if action[3] == 1:
                        power_coms0_2 = (Vars.power_trans_full_red* (wait_packetSize0_2/available_trans0_2) + Vars.power_idle_red * (1-wait_packetSize0_2/trans_rate_list_user2[0]))*(1-transition_time_ratio0)/1000  #两部分功率：传输过程中 + 空闲状态
                    elif action[3] == 2:
                        power_coms0_2 = (Vars.power_trans_full_black * (wait_packetSize0_2 / available_trans0_2) + Vars.power_idle_black * (1 - wait_packetSize0_2 / trans_rate_list_user2[0])) * (1 - transition_time_ratio0) / 1000
                    elif action[3] == 3:
                        power_coms0_2 = (Vars.power_trans_full_green * (wait_packetSize0_2 / available_trans0_2) + Vars.power_idle_green * (1 - wait_packetSize0_2 / trans_rate_list_user2[0])) * (1 - transition_time_ratio0) / 1000
                    energy_sum += power_coms0_2 #将功率加进来
                    trans_flag0_2 = True
                    first_data_index0_2 = np.where(arrival_data_matrix0[:,2] !=0)
                    if first_data_index0_2 and len(first_data_index0_2[0]) > 0:
                        tran_no0_2 = first_data_index0_2[0][0]
                    else:
                        tran_no0_2 = 0
                        print('first_data_index0_2', first_data_index0_2)
                    Vars.start_trans_no0_2 = tran_no0_2
                    sum_trans0_2 = 0#当前1ms内的已经处理的能力
                    while trans_flag0_2 == True and tran_no0_2 < len(arrival_data_matrix0) and sum_trans0_2 < wait_packetSize0_2:#当前时刻已经有一部分处理的数据了，但是由于大的判断条件为wait_packetSize0 < available_trans0，所以这里要判断一下sum_trans0 < wait_packetSize0，说明当前1ms内还有一部分没有处理完的waitpacketsize
                        if arrival_data_matrix0[tran_no0_2][2] > 0:
                            if sum_trans0_2 + arrival_data_matrix0[tran_no0_2][2] > available_trans0_2:
                                arrival_data_matrix0[tran_no0_2][2] = arrival_data_matrix0[tran_no0_2][2] -(available_trans0_2-sum_trans0_2)
                                tran_no0_2 = False
                            else:
                                sum_trans0_2 = sum_trans0_2 +  arrival_data_matrix0[tran_no0_2][2]
                                arrival_data_matrix0[tran_no0_2][2] = 0
                                if arrival_data_matrix0[tran_no0_2][1] == 0 and arrival_data_matrix0[tran_no0_2][3] == 0 and arrival_data_matrix0[tran_no0_2][5] == 0:
                                    #当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix0[tran_no0_2][7] = Vars.time#记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index0_2 = np.where(arrival_data_matrix0[:, 2] != 0)
                                if first_data_index0_2 and len(first_data_index0_2[0]) > 0:
                                    tran_no0_2 = first_data_index0_2[0][0]
                                else:
                                    tran_no0_2 = 0
                                    print('first_data_index0_2', first_data_index0_2)
                                Vars.start_trans_no0_2 = tran_no0_2#转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix0[tran_no0_2][0] = arrival_data_matrix0[tran_no0_2][2] + \
                                                          arrival_data_matrix0[tran_no0_2][3] + \
                                                          arrival_data_matrix0[tran_no0_2][5]
                #for user3
                if wait_packetSize0_3 >= available_trans0_3 : #说明使用当前全部的1ms才能刚好处理或处理不完data
                    #先判断zooming的大小
                    if action[3] == 1: #当前处于red圆
                        #计算energy
                        power_coms0_3 = Vars.power_trans_full_red * (1-transition_time_ratio0)
                    elif action[3] == 2: #当前处于black圆
                        power_coms0_3 = Vars.power_trans_full_black * (1 - transition_time_ratio0)
                    elif action[3] == 3: #当前处于green圆
                        power_coms0_3 = Vars.power_trans_full_green  * (1 - transition_time_ratio0)
                    energy_sum += power_coms0_3  #将当前时刻处理数据消耗的功耗加到总功耗中，用于计算reward
                    trans_flag0_3  = True #用True标记一直传输
                    first_data_index0_3 = np.where(arrival_data_matrix0[:,3] !=0)
                    if first_data_index0_3 and len(first_data_index0_3[0] > 0):#当整列数都为0时，说明当前这个用户在当前的BS没有连接且没有数据
                        tran_no0_3 = first_data_index0_3[0][0]
                    else:#当长度不为0时，说明有数据，则tran_no0_2为第一个不为0的数的行index
                        print('first_data_index0_3', first_data_index0_3)
                        tran_no0_3 = 0
                    Vars.start_trans_no0_3 = tran_no0_3
                    sum_trans0_3 = 0 #指当前1ms内已经处理完的数据
                    while trans_flag0_3  == True and tran_no0_3 < len(arrival_data_matrix0):#还在处理数据切合法
                        if arrival_data_matrix0[tran_no0_3][3]>0: #user2有等待数据#指当前arrival_data_matrix0行的数据还没处理完，如第1行100，处理完50，还剩50时
                            if sum_trans0_3  + arrival_data_matrix0[tran_no0_3][3] > available_trans0_3 :# sum_trans0指当前1ms内已经消耗的处理能力，arrival_data_matrix0[trans_no][0]指当前行的剩余数据
                                #如果当前1ms消耗的处理能力+matrix当前行的剩余数据>处理能力，说明当前的1ms处理不完arrival_data_matrix0[trans_no][0]的数据
                                arrival_data_matrix0[tran_no0_3][3] = arrival_data_matrix0[tran_no0_3][3] - (available_trans0_3  - sum_trans0_3 )
                                trans_flag0_3  = False#当前1ms已经消耗完,这一秒已经过完了，直接退出这1ms，进入下1ms，下一次大循环了
                            else:#如果sum_trans0 + arrival_data_matrix0[trans_no][0] 小于 available_trans0，说明当前1ms内剩余的处理能力可以把当前的arrival_data_matrix0[trans_no][1]的数据处理完
                                sum_trans0_3  = sum_trans0_3  + arrival_data_matrix0[tran_no0_3][3]
                                arrival_data_matrix0[tran_no0_3][3] = 0 #当当前行的数据被处理完
                                if arrival_data_matrix0[tran_no0_3][2] == 0 and arrival_data_matrix0[tran_no0_3][3] == 0 and arrival_data_matrix0[tran_no0_3][5] == 0:
                                    #当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix0[tran_no0_3][7] = Vars.time#记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index0_3 = np.where(arrival_data_matrix0[:, 3] != 0)
                                if first_data_index0_3 and len(first_data_index0_3[0] > 0):
                                    tran_no0_3 = first_data_index0_3[0][0]
                                else:
                                    print('first_data_index0_3', first_data_index0_3)
                                    tran_no0_3 = 0
                                Vars.start_trans_no0_3 = tran_no0_3#转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix0[tran_no0_3][0] = arrival_data_matrix0[tran_no0_3][2] + \
                                                          arrival_data_matrix0[tran_no0_3][3] + \
                                                          arrival_data_matrix0[tran_no0_3][5]

                        #在else中没有trans_flag0设置为false（因为当前的1ms有能力处理完当前数据，而我们不确定当前1ms是否还可以继续处理数据，因此循环继续处理）
                elif  (wait_packetSize0_3 > 0) and (wait_packetSize0_3 < available_trans0_3): #说明当前的1ms完全可以处理完当前的数据，且处理完后因为没有立刻转入SM状态，因此在activate中有一段空闲时刻
                    if action[3] == 1:
                        power_coms0_3 = (Vars.power_trans_full_red* (wait_packetSize0_3/available_trans0_3) + Vars.power_idle_red * (1-wait_packetSize0_3/trans_rate_list_user3[0]))*(1-transition_time_ratio0)/1000  #两部分功率：传输过程中 + 空闲状态
                    elif action[3] == 2:
                        power_coms0_3 = (Vars.power_trans_full_black * (wait_packetSize0_3 / available_trans0_3) + Vars.power_idle_black * (1 - wait_packetSize0_3 / trans_rate_list_user3[0])) * (1 - transition_time_ratio0) / 1000
                    elif action[3] == 3:
                        power_coms0_3 = (Vars.power_trans_full_green * (wait_packetSize0_3 / available_trans0_3) + Vars.power_idle_green * (1 - wait_packetSize0_3 / trans_rate_list_user3[0])) * (1 - transition_time_ratio0) / 1000
                    energy_sum += power_coms0_3 #将功率加进来
                    trans_flag0_3 = True
                    first_data_index0_3 = np.where(arrival_data_matrix0[:, 3] != 0)
                    if first_data_index0_3 and len(first_data_index0_3[0] > 0):
                        tran_no0_3 = first_data_index0_3[0][0]
                    else:
                        print('first_data_index0_3', first_data_index0_3)
                        tran_no0_3 = 0
                    Vars.start_trans_no0_3 = tran_no0_3
                    sum_trans0_3 = 0#当前1ms内的已经处理的能力
                    while trans_flag0_3 == True and tran_no0_3 < len(arrival_data_matrix0) and sum_trans0_3 < wait_packetSize0_3:#当前时刻已经有一部分处理的数据了，但是由于大的判断条件为wait_packetSize0 < available_trans0，所以这里要判断一下sum_trans0 < wait_packetSize0，说明当前1ms内还有一部分没有处理完的waitpacketsize
                        if arrival_data_matrix0[tran_no0_3][3] > 0:
                            if sum_trans0_3 + arrival_data_matrix0[tran_no0_3][3] > available_trans0_3:
                                arrival_data_matrix0[tran_no0_3][3] = arrival_data_matrix0[tran_no0_3][3] -(available_trans0_3-sum_trans0_3)
                                tran_no0_3 = False
                            else:
                                sum_trans0_3 = sum_trans0_3 +  arrival_data_matrix0[tran_no0_3][3]
                                arrival_data_matrix0[tran_no0_3][3] = 0
                                if arrival_data_matrix0[tran_no0_3][2] == 0 and arrival_data_matrix0[tran_no0_3][3] == 0 and arrival_data_matrix0[tran_no0_3][5] == 0:
                                    #当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix0[tran_no0_3][7] = Vars.time#记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index0_3 = np.where(arrival_data_matrix0[:, 3] != 0)
                                if first_data_index0_3 and len(first_data_index0_3[0] > 0):
                                    tran_no0_3 = first_data_index0_3[0][0]
                                else:
                                    print('first_data_index0_3', first_data_index0_3)
                                    tran_no0_3 = 0
                                Vars.start_trans_no0_3 = tran_no0_3#转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix0[tran_no0_3][0] = arrival_data_matrix0[tran_no0_3][2] + arrival_data_matrix0[tran_no0_3][3] + arrival_data_matrix0[tran_no0_3][5]

                #for user5
                if wait_packetSize0_5 >= available_trans0_5 : #说明使用当前全部的1ms才能刚好处理或处理不完data
                    #先判断zooming的大小
                    if action[3] == 1: #当前处于red圆
                        #计算energy
                        power_coms0_5 = Vars.power_trans_full_red * (1-transition_time_ratio0)
                    elif action[3] == 2: #当前处于black圆
                        power_coms0_5 = Vars.power_trans_full_black * (1 - transition_time_ratio0)
                    elif action[3] == 3: #当前处于green圆
                        power_coms0_5 = Vars.power_trans_full_green  * (1 - transition_time_ratio0)
                    energy_sum += power_coms0_5  #将当前时刻处理数据消耗的功耗加到总功耗中，用于计算reward
                    trans_flag0_5  = True #用True标记一直传输
                    first_data_index0_5 = np.where(arrival_data_matrix0[:,5] !=0)
                    if first_data_index0_5 and len(first_data_index0_5[0] >0):
                        tran_no0_5 = first_data_index0_5[0][0]
                    else:
                        print('first_data_index0_5', first_data_index0_5)
                        tran_no0_5 = 0
                    Vars.start_trans_no0_5 = tran_no0_5
                    sum_trans0_5 = 0 #指当前1ms内已经处理完的数据
                    while trans_flag0_5  == True and tran_no0_5 < len(arrival_data_matrix0):#还在处理数据切合法
                        if arrival_data_matrix0[tran_no0_5][5]>0: #user2有等待数据#指当前arrival_data_matrix0行的数据还没处理完，如第1行100，处理完50，还剩50时
                            if sum_trans0_5  + arrival_data_matrix0[tran_no0_5][5] > available_trans0_5 :# sum_trans0指当前1ms内已经消耗的处理能力，arrival_data_matrix0[trans_no][0]指当前行的剩余数据
                                #如果当前1ms消耗的处理能力+matrix当前行的剩余数据>处理能力，说明当前的1ms处理不完arrival_data_matrix0[trans_no][0]的数据
                                arrival_data_matrix0[tran_no0_5][5] = arrival_data_matrix0[tran_no0_5][5] - (available_trans0_5  - sum_trans0_5 )
                                trans_flag0_5  = False#当前1ms已经消耗完,这一秒已经过完了，直接退出这1ms，进入下1ms，下一次大循环了
                            else:#如果sum_trans0 + arrival_data_matrix0[trans_no][0] 小于 available_trans0，说明当前1ms内剩余的处理能力可以把当前的arrival_data_matrix0[trans_no][1]的数据处理完
                                sum_trans0_5  = sum_trans0_5  + arrival_data_matrix0[tran_no0_5][5]
                                arrival_data_matrix0[tran_no0_5][5] = 0 #当当前行的数据被处理完
                                if arrival_data_matrix0[tran_no0_5][2] == 0 and arrival_data_matrix0[tran_no0_5][3] == 0 and arrival_data_matrix0[tran_no0_5][5] == 0:
                                    #当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix0[tran_no0_5][7] = Vars.time#记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index0_5 = np.where(arrival_data_matrix0[:, 5] != 0)
                                if first_data_index0_5 and len(first_data_index0_5[0] > 0):
                                    tran_no0_5 = first_data_index0_5[0][0]
                                else:
                                    print('first_data_index0_5', first_data_index0_5)
                                    tran_no0_5 = 0
                                Vars.start_trans_no0_5 = tran_no0_5#转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix0[tran_no0_5][0] = arrival_data_matrix0[tran_no0_5][2] + \
                                                          arrival_data_matrix0[tran_no0_5][3] + \
                                                          arrival_data_matrix0[tran_no0_5][5]
                #在else中没有trans_flag0设置为false（因为当前的1ms有能力处理完当前数据，而我们不确定当前1ms是否还可以继续处理数据，因此循环继续处理）
                elif  (wait_packetSize0_5 > 0) and (wait_packetSize0_5 < available_trans0_5): #说明当前的1ms完全可以处理完当前的数据，且处理完后因为没有立刻转入SM状态，因此在activate中有一段空闲时刻
                    if action[3] == 1:
                        power_coms0_5 = (Vars.power_trans_full_red* (wait_packetSize0_5/available_trans0_5) + Vars.power_idle_red * (1-wait_packetSize0_5/trans_rate_list_user5[0]))*(1-transition_time_ratio0)/1000  #两部分功率：传输过程中 + 空闲状态
                    elif action[3] == 2:
                        power_coms0_5 = (Vars.power_trans_full_black * (wait_packetSize0_5 / available_trans0_5) + Vars.power_idle_black * (1 - wait_packetSize0_5 / trans_rate_list_user5[0])) * (1 - transition_time_ratio0) / 1000
                    elif action[3] == 3:
                        power_coms0_5 = (Vars.power_trans_full_green * (wait_packetSize0_5 / available_trans0_5) + Vars.power_idle_green * (1 - wait_packetSize0_5 / trans_rate_list_user5[0])) * (1 - transition_time_ratio0) / 1000
                    energy_sum += power_coms0_5 #将功率加进来
                    trans_flag0_5 = True
                    first_data_index0_5 = np.where(arrival_data_matrix0[:, 5] != 0)
                    if first_data_index0_5 and len(first_data_index0_5[0] >0):
                        tran_no0_5 = first_data_index0_5[0][0]
                    else:
                        print('first_data_index0_5', first_data_index0_5)
                        tran_no0_5 = 0
                    Vars.start_trans_no0_5 = tran_no0_5
                    sum_trans0_5 = 0#当前1ms内的已经处理的能力
                    while trans_flag0_5 == True and tran_no0_5 < len(arrival_data_matrix0) and sum_trans0_5 < wait_packetSize0_5:#当前时刻已经有一部分处理的数据了，但是由于大的判断条件为wait_packetSize0 < available_trans0，所以这里要判断一下sum_trans0 < wait_packetSize0，说明当前1ms内还有一部分没有处理完的waitpacketsize
                        if arrival_data_matrix0[tran_no0_5][5] > 0:
                            if sum_trans0_5 + arrival_data_matrix0[tran_no0_5][5] > available_trans0_5:
                                arrival_data_matrix0[tran_no0_5][5] = arrival_data_matrix0[tran_no0_5][5] -(available_trans0_5-sum_trans0_5)
                                tran_no0_5 = False
                            else:
                                sum_trans0_5 = sum_trans0_5 +  arrival_data_matrix0[tran_no0_5][5]
                                arrival_data_matrix0[tran_no0_5][5] = 0
                                if arrival_data_matrix0[tran_no0_5][2] == 0 and arrival_data_matrix0[tran_no0_5][3] == 0 and arrival_data_matrix0[tran_no0_5][5] == 0:
                                    #当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix0[tran_no0_5][7] = Vars.time#记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index0_5 = np.where(arrival_data_matrix0[:, 5] != 0)
                                if first_data_index0_5 and len(first_data_index0_5[0] > 0):
                                    tran_no0_5 = first_data_index0_5[0][0]
                                else:
                                    print('first_data_index0_5', first_data_index0_5)
                                    tran_no0_5 = 0
                                Vars.start_trans_no0_5 = tran_no0_5#转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix0[tran_no0_5][0] = arrival_data_matrix0[tran_no0_5][2] + arrival_data_matrix0[tran_no0_5][3] + arrival_data_matrix0[tran_no0_5][5]
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
        # 按照基站，先一个基站一个基站的处理，action的第1-3列记录了BS0model，BS1model，BS2model（列index0，1，2）

    # print('BS1')

    # BS1
    if transition_time1 != 0:  # 说明当前处于work model转换阶段
        if 0 < transition_time1 < 1:  # 说明当前处于模过渡阶段，且过渡时间transition_time_ratio0小于1ms
            transition_time_ratio1 = transition_time1
            transition_time1 = 0
        elif transition_time1 > 1:  # 当前处于模式过渡阶段且过渡时间大于1ms
            transition_time1 = transition_time1 - 1
    else:
        if BSmodel1 == 0:  # activate
            # 连接了user1/2/5
            available_trans1_1 = trans_rate_list_user1[1] * (1 - transition_time_ratio1)  # user1
            available_trans1_2 = trans_rate_list_user2[1] * (1 - transition_time_ratio1)  # user2
            available_trans1_5 = trans_rate_list_user5[1] * (1 - transition_time_ratio1)  # user5

            if wait_packetSize1 > 0:

                # for user1
                if wait_packetSize1_1 >= available_trans1_1:  # 说明使用当前全部的1ms才能刚好处理或处理不完data
                    # 先判断zooming的大小
                    if action[4] == 1:  # 当前处于red圆
                        # 计算energy
                        power_coms1_1 = Vars.power_trans_full_red * (1 - transition_time_ratio1)
                    elif action[4] == 2:  # 当前处于black圆
                        power_coms1_1 = Vars.power_trans_full_black * (1 - transition_time_ratio1)
                    elif action[4] == 3:  # 当前处于green圆
                        power_coms1_1 = Vars.power_trans_full_green * (1 - transition_time_ratio1)
                    energy_sum += power_coms1_1  # 将当前时刻处理数据消耗的功耗加到总功耗中，用于计算reward
                    trans_flag1_1 = True  # 用True标记一直传输
                    first_data_index1_1 = np.where(arrival_data_matrix1[:, 1] != 0)
                    if first_data_index1_1 and len(first_data_index1_1[0] > 0):
                        tran_no1_1 = first_data_index1_1[0][0]
                    else:
                        print('first_data_index1_1', first_data_index1_1)
                        tran_no1_1 = 0
                    Vars.start_trans_no1_1 = tran_no1_1
                    sum_trans1_1 = 0  # 指当前1ms内已经处理完的数据
                    while trans_flag1_1 == True and tran_no1_1 < len(arrival_data_matrix1):  # 还在处理数据切合法
                        if arrival_data_matrix1[tran_no1_1][1] > 0:  # user2有等待数据#指当前arrival_data_matrix1行的数据还没处理完，如第1行100，处理完50，还剩50时
                            if sum_trans1_1 + arrival_data_matrix1[tran_no1_1][1] > available_trans1_1:  # sum_trans0指当前1ms内已经消耗的处理能力，arrival_data_matrix1[trans_no][0]指当前行的剩余数据
                                # 如果当前1ms消耗的处理能力+matrix当前行的剩余数据>处理能力，说明当前的1ms处理不完arrival_data_matrix1arrival_data_matrix1[trans_no][0]的数据
                                arrival_data_matrix1[tran_no1_1][1] = arrival_data_matrix1[tran_no1_1][1] - (available_trans1_1 - sum_trans1_1)
                                trans_flag1_1 = False  # 当前1ms已经消耗完,这一秒已经过完了，直接退出这1ms，进入下1ms，下一次大循环了
                            else:  # 如果sum_trans0 + arrival_data_matrix1[trans_no][0] 小于 available_trans0，说明当前1ms内剩余的处理能力可以把当前的arrival_data_matrix1[trans_no][1]的数据处理完
                                sum_trans1_1 = sum_trans1_1 + arrival_data_matrix1[tran_no1_1][1]
                                arrival_data_matrix1[tran_no1_1][1] = 0  # 当当前行的数据被处理完
                                if arrival_data_matrix1[tran_no1_1][1] == 0 and arrival_data_matrix1[tran_no1_1][2] == 0 and arrival_data_matrix1[tran_no1_1][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix1[tran_no1_1][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index1_1 = np.where(arrival_data_matrix1[:, 1] != 0)
                                if first_data_index1_1 and len(first_data_index1_1[0] > 0):
                                    tran_no1_1 = first_data_index1_1[0][0]
                                else:
                                    print('first_data_index1_1', first_data_index1_1)
                                    tran_no1_1 = 0
                                Vars.start_trans_no1_1 = tran_no1_1  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix1[tran_no1_1][0] = arrival_data_matrix1[tran_no1_1][1] + \
                                                          arrival_data_matrix1[tran_no1_1][2] + \
                                                          arrival_data_matrix1[tran_no1_1][5]
                        # 在else中没有trans_flag0设置为false（因为当前的1ms有能力处理完当前数据，而我们不确定当前1ms是否还可以继续处理数据，因此循环继续处理）
                elif (wait_packetSize1_1 > 0) and (wait_packetSize1_1 < available_trans1_1):  # 说明当前的1ms完全可以处理完当前的数据，且处理完后因为没有立刻转入SM状态，因此在activate中有一段空闲时刻
                    if action[4] == 1:
                        power_coms1_1 = (Vars.power_trans_full_red * (wait_packetSize1_1 / available_trans1_1) + Vars.power_idle_red * (1 - wait_packetSize1_1 / trans_rate_list_user1[1])) * (1 - transition_time_ratio1) / 1000  # 两部分功率：传输过程中 + 空闲状态
                    elif action[4] == 2:
                        power_coms1_1 = (Vars.power_trans_full_black * (wait_packetSize1_1 / available_trans1_1) + Vars.power_idle_black * (1 - wait_packetSize1_1 / trans_rate_list_user1[1])) * (1 - transition_time_ratio1) / 1000
                    elif action[4] == 3:
                        power_coms1_1 = (Vars.power_trans_full_green * (wait_packetSize1_1 / available_trans1_1) + Vars.power_idle_green * (1 - wait_packetSize1_1 / trans_rate_list_user1[1])) * (1 - transition_time_ratio1) / 1000
                    energy_sum += power_coms1_1  # 将功率加进来
                    trans_flag1_1 = True
                    first_data_index1_1 = np.where(arrival_data_matrix1[:, 1] != 0)
                    if first_data_index1_1 and len(first_data_index1_1[0] > 0):
                        tran_no1_1 = first_data_index1_1[0][0]
                    else:
                        print('first_data_index1_1', first_data_index1_1)
                        tran_no1_1 = 0
                    Vars.start_trans_no1_1 = tran_no1_1
                    sum_trans1_1 = 0  # 当前1ms内的已经处理的能力
                    while trans_flag1_1 == True and tran_no1_1 < len(arrival_data_matrix1) and sum_trans1_1 < wait_packetSize1_1:  # 当前时刻已经有一部分处理的数据了，但是由于大的判断条件为wait_packetSize0 < available_trans0，所以这里要判断一下sum_trans0 < wait_packetSize0，说明当前1ms内还有一部分没有处理完的waitpacketsize
                        if arrival_data_matrix1[tran_no1_1][1] > 0:
                            if sum_trans1_1 + arrival_data_matrix1[tran_no1_1][1] > available_trans1_1:
                                arrival_data_matrix1[tran_no1_1][1] = arrival_data_matrix1[tran_no1_1][1] - (available_trans1_1 - sum_trans1_1)
                                tran_no1_1 = False
                            else:
                                sum_trans1_1 = sum_trans1_1 + arrival_data_matrix1[tran_no1_1][1]
                                arrival_data_matrix1[tran_no1_1][1] = 0
                                if arrival_data_matrix1[tran_no1_1][1] == 0 and arrival_data_matrix1[tran_no1_1][2] == 0 and arrival_data_matrix1[tran_no1_1][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix1[tran_no1_1][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index1_1 = np.where(arrival_data_matrix1[:, 1] != 0)
                                if first_data_index1_1 and len(first_data_index1_1[0] > 0):
                                    tran_no1_1 = first_data_index1_1[0][0]
                                else:
                                    print('first_data_index1_1', first_data_index1_1)
                                    tran_no1_1 = 0
                                Vars.start_trans_no1_1 = tran_no1_1  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix1[tran_no1_1][0] = arrival_data_matrix1[tran_no1_1][1] + \
                                                          arrival_data_matrix1[tran_no1_1][2] + \
                                                          arrival_data_matrix1[tran_no1_1][5]
                # for user2
                if wait_packetSize1_2 >= available_trans1_2:  # 说明使用当前全部的1ms才能刚好处理或处理不完data
                    # 先判断zooming的大小
                    if action[4] == 1:  # 当前处于red圆
                        # 计算energy
                        power_coms1_2 = Vars.power_trans_full_red * (1 - transition_time_ratio1)
                    elif action[4] == 2:  # 当前处于black圆
                        power_coms1_2 = Vars.power_trans_full_black * (1 - transition_time_ratio1)
                    elif action[4] == 3:  # 当前处于green圆
                        power_coms1_2 = Vars.power_trans_full_green * (1 - transition_time_ratio1)
                    energy_sum += power_coms1_2  # 将当前时刻处理数据消耗的功耗加到总功耗中，用于计算reward
                    trans_flag1_2 = True  # 用True标记一直传输
                    first_data_index1_2 = np.where(arrival_data_matrix1[:, 2] != 0)
                    if first_data_index1_2 and len(first_data_index1_2[0] > 0):
                        tran_no1_2 = first_data_index1_2[0][0]
                    else:
                        print('first_data_index1_2', first_data_index1_2)
                        tran_no1_2 = 0
                    Vars.start_trans_no1_2 = tran_no1_2
                    sum_trans1_2 = 0  # 指当前1ms内已经处理完的数据
                    while trans_flag1_2 == True and tran_no1_2 < len(arrival_data_matrix1):  # 还在处理数据切合法
                        if arrival_data_matrix1[tran_no1_2][2] > 0:  # user2有等待数据#指当前arrival_data_matrix1行的数据还没处理完，如第1行100，处理完50，还剩50时
                            if sum_trans1_2 + arrival_data_matrix1[tran_no1_2][2] > available_trans1_2:  # sum_trans0指当前1ms内已经消耗的处理能力，arrival_data_matrix1[trans_no][0]指当前行的剩余数据
                                # 如果当前1ms消耗的处理能力+matrix当前行的剩余数据>处理能力，说明当前的1ms处理不完arrival_data_matrix1[trans_no][0]的数据
                                arrival_data_matrix1[tran_no1_2][2] = arrival_data_matrix1[tran_no1_2][2] - (available_trans1_2 - sum_trans1_2)
                                trans_flag1_2 = False  # 当前1ms已经消耗完,这一秒已经过完了，直接退出这1ms，进入下1ms，下一次大循环了
                            else:  # 如果sum_trans0 + arrival_data_matrix1[trans_no][0] 小于 available_trans0，说明当前1ms内剩余的处理能力可以把当前的arrival_data_matrix1[trans_no][1]的数据处理完
                                sum_trans1_2 = sum_trans1_2 + arrival_data_matrix1[tran_no1_2][2]
                                arrival_data_matrix1[tran_no1_2][2] = 0  # 当当前行的数据被处理完
                                if arrival_data_matrix1[trans_no1_2][1] == 0 and arrival_data_matrix1[trans_no1_2][2] == 0 and arrival_data_matrix1[trans_no1_2][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix1[trans_no1_2][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index1_2 = np.where(arrival_data_matrix1[:, 2] != 0)
                                if first_data_index1_2 and len(first_data_index1_2[0] > 0):
                                    tran_no1_2 = first_data_index1_2[0][0]
                                else:
                                    print('first_data_index1_2', first_data_index1_2)
                                    tran_no1_2 = 0
                                Vars.start_trans_no1_2 = tran_no1_2  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix1[tran_no1_2][0] = arrival_data_matrix1[tran_no1_2][1] + \
                                                          arrival_data_matrix1[tran_no1_2][2] + \
                                                          arrival_data_matrix1[tran_no1_2][5]
                        # 在else中没有trans_flag0设置为false（因为当前的1ms有能力处理完当前数据，而我们不确定当前1ms是否还可以继续处理数据，因此循环继续处理）
                elif (wait_packetSize1_2 > 0) and (wait_packetSize1_2 < available_trans1_2): # 说明当前的1ms完全可以处理完当前的数据，且处理完后因为没有立刻转入SM状态，因此在activate中有一段空闲时刻
                    if action[4] == 1:
                        power_coms1_2 = (Vars.power_trans_full_red * (wait_packetSize1_2 / available_trans1_2) + Vars.power_idle_red * (1 - wait_packetSize1_2 / trans_rate_list_user2[1])) * (1 - transition_time_ratio1) / 1000  # 两部分功率：传输过程中 + 空闲状态
                    elif action[4] == 2:
                        power_coms1_2 = (Vars.power_trans_full_black * (wait_packetSize1_2 / available_trans1_2) + Vars.power_idle_black * (1 - wait_packetSize1_2 / trans_rate_list_user2[1])) * (1 - transition_time_ratio1) / 1000
                    elif action[4] == 3:
                        power_coms1_2 = (Vars.power_trans_full_green * (wait_packetSize1_2 / available_trans1_2) + Vars.power_idle_green * (1 - wait_packetSize1_2 / trans_rate_list_user2[1])) * (1 - transition_time_ratio1) / 1000
                    energy_sum += power_coms1_2  # 将功率加进来
                    trans_flag1_2 = True
                    first_data_index1_2 = np.where(arrival_data_matrix1[:, 2] != 0)
                    if first_data_index1_2 and len(first_data_index1_2[0] > 0):
                        tran_no1_2 = first_data_index1_2[0][0]
                    else:
                        print('first_data_index1_2', first_data_index1_2)
                        tran_no1_2 = 0
                    Vars.start_trans_no1_2 = tran_no1_2
                    sum_trans1_2 = 0  # 当前1ms内的已经处理的能力
                    while trans_flag1_2 == True and tran_no1_2 < len(arrival_data_matrix1) and sum_trans1_2 < wait_packetSize1_2:  # 当前时刻已经有一部分处理的数据了，但是由于大的判断条件为wait_packetSize0 < available_trans0，所以这里要判断一下sum_trans0 < wait_packetSize0，说明当前1ms内还有一部分没有处理完的waitpacketsize
                        if arrival_data_matrix1[tran_no1_2][2] > 0:
                            if sum_trans1_2 + arrival_data_matrix1[tran_no1_2][2] > available_trans1_2:
                                arrival_data_matrix1[tran_no1_2][2] = arrival_data_matrix1[tran_no1_2][2] - (available_trans1_2 - sum_trans1_2)
                                tran_no1_2 = False
                            else:
                                sum_trans1_2 = sum_trans1_2 + arrival_data_matrix1[tran_no1_2][2]
                                arrival_data_matrix1[tran_no1_2][2] = 0
                                if arrival_data_matrix1[trans_no1_2][1] == 0 and arrival_data_matrix1[trans_no1_2][2] == 0 and arrival_data_matrix1[trans_no1_2][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix1[trans_no1_2][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index1_2 = np.where(arrival_data_matrix1[:, 2] != 0)
                                if first_data_index1_2 and len(first_data_index1_2[0] > 0):
                                    tran_no1_2 = first_data_index1_2[0][0]
                                else:
                                    print('first_data_index1_2', first_data_index1_2)
                                    tran_no1_2 = 0
                                Vars.start_trans_no1_2 = tran_no1_2  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix1[tran_no1_2][0] = arrival_data_matrix1[tran_no1_2][1] + \
                                                          arrival_data_matrix1[tran_no1_2][2] + \
                                                          arrival_data_matrix1[tran_no1_2][5]
                # for user5
                if wait_packetSize1_5 >= available_trans1_5:  # 说明使用当前全部的1ms才能刚好处理或处理不完data
                    # 先判断zooming的大小
                    if action[4] == 1:  # 当前处于red圆
                        # 计算energy
                        power_coms1_5 = Vars.power_trans_full_red * (1 - transition_time_ratio1)
                    elif action[4] == 2:  # 当前处于black圆
                        power_coms1_5 = Vars.power_trans_full_black * (1 - transition_time_ratio1)
                    elif action[4] == 3:  # 当前处于green圆
                        power_coms1_5 = Vars.power_trans_full_green * (1 - transition_time_ratio1)
                    energy_sum += power_coms1_5  # 将当前时刻处理数据消耗的功耗加到总功耗中，用于计算reward
                    trans_flag1_5 = True  # 用True标记一直传输
                    first_data_index1_5 = np.where(arrival_data_matrix1[:, 5] != 0)
                    if first_data_index1_5 and len(first_data_index1_5[0] > 0):
                        tran_no1_5 = first_data_index1_5[0][0]
                    else:
                        print('first_data_index1_5', first_data_index1_5)
                        tran_no1_5 = 0
                    Vars.start_trans_no1_5 = tran_no1_5
                    sum_trans1_5 = 0  # 指当前1ms内已经处理完的数据
                    while trans_flag1_5 == True and tran_no1_5 < len(arrival_data_matrix1):  # 还在处理数据切合法
                        if arrival_data_matrix1[tran_no1_5][5] > 0:  # user2有等待数据#指当前arrival_data_matrix1行的数据还没处理完，如第1行100，处理完50，还剩50时
                            if sum_trans1_5 + arrival_data_matrix1[tran_no1_5][5] > available_trans1_5:  # sum_trans0指当前1ms内已经消耗的处理能力，arrival_data_matrix1[trans_no][0]指当前行的剩余数据
                                # 如果当前1ms消耗的处理能力+matrix当前行的剩余数据>处理能力，说明当前的1ms处理不完arrival_data_matrix1[trans_no][0]的数据
                                arrival_data_matrix1[tran_no1_5][5] = arrival_data_matrix1[tran_no1_5][5] - (available_trans1_5 - sum_trans1_5)
                                trans_flag1_5 = False  # 当前1ms已经消耗完,这一秒已经过完了，直接退出这1ms，进入下1ms，下一次大循环了
                            else:  # 如果sum_trans0 + arrival_data_matrix1[trans_no][0] 小于 available_trans0，说明当前1ms内剩余的处理能力可以把当前的arrival_data_matrix1[trans_no][1]的数据处理完
                                sum_trans1_5 = sum_trans1_5 + arrival_data_matrix1[tran_no1_5][5]
                                arrival_data_matrix1[tran_no1_5][5] = 0  # 当当前行的数据被处理完
                                if arrival_data_matrix1[tran_no1_5][1] == 0 and arrival_data_matrix1[tran_no1_5][2] == 0 and arrival_data_matrix1[tran_no1_5][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix1[tran_no1_5][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index1_5 = np.where(arrival_data_matrix1[:, 5] != 0)
                                if first_data_index1_5 and len(first_data_index1_5[0] > 0):
                                    tran_no1_5 = first_data_index1_5[0][0]
                                else:
                                    print('first_data_index1_5', first_data_index1_5)
                                    tran_no1_5 = 0
                                Vars.start_trans_no1_5 = tran_no1_5  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix1[tran_no1_5][0] = arrival_data_matrix1[tran_no1_5][1] + \
                                                          arrival_data_matrix1[tran_no1_5][2] + \
                                                          arrival_data_matrix1[tran_no1_5][5]
                        # 在else中没有trans_flag0设置为false（因为当前的1ms有能力处理完当前数据，而我们不确定当前1ms是否还可以继续处理数据，因此循环继续处理）
                elif (wait_packetSize1_5 > 0) and (wait_packetSize1_5 < available_trans1_5):  # 说明当前的1ms完全可以处理完当前的数据，且处理完后因为没有立刻转入SM状态，因此在activate中有一段空闲时刻
                    if action[4] == 1:
                        power_coms1_5 = (Vars.power_trans_full_red * (wait_packetSize1_5 / available_trans1_5) + Vars.power_idle_red * (1 - wait_packetSize1_5 / trans_rate_list_user5[1])) * (1 - transition_time_ratio1) / 1000  # 两部分功率：传输过程中 + 空闲状态
                    elif action[4] == 2:
                        power_coms1_5 = (Vars.power_trans_full_black * (wait_packetSize1_5 / available_trans1_5) + Vars.power_idle_black * (1 - wait_packetSize1_5 / trans_rate_list_user5[1])) * (1 - transition_time_ratio1) / 1000
                    elif action[4] == 3:
                        power_coms1_5 = (Vars.power_trans_full_green * (wait_packetSize1_5 / available_trans1_5) + Vars.power_idle_green * (1 - wait_packetSize1_5 / trans_rate_list_user5[1])) * (1 - transition_time_ratio1) / 1000
                    energy_sum += power_coms1_5  # 将功率加进来
                    trans_flag1_5 = True
                    first_data_index1_5 = np.where(arrival_data_matrix1[:, 5] != 0)
                    if first_data_index1_5 and len(first_data_index1_5[0] > 0):
                        tran_no1_5 = first_data_index1_5[0][0]
                    else:
                        print('first_data_index1_5', first_data_index1_5)
                        tran_no1_5 = 0
                    Vars.start_trans_no1_5 = tran_no1_5
                    sum_trans1_5 = 0  # 当前1ms内的已经处理的能力
                    while trans_flag1_5 == True and tran_no1_5 < len(arrival_data_matrix1) and sum_trans1_5 < wait_packetSize1_5:  # 当前时刻已经有一部分处理的数据了，但是由于大的判断条件为wait_packetSize0 < available_trans0，所以这里要判断一下sum_trans0 < wait_packetSize0，说明当前1ms内还有一部分没有处理完的waitpacketsize
                        if arrival_data_matrix1[tran_no1_5][5] > 0:
                            if sum_trans1_5 + arrival_data_matrix1[tran_no1_5][5] > available_trans1_5:
                                arrival_data_matrix1[tran_no1_5][5] = arrival_data_matrix1[tran_no1_5][5] - (available_trans1_5 - sum_trans1_5)
                                tran_no1_5 = False
                            else:
                                sum_trans1_5 = sum_trans1_5 + arrival_data_matrix1[tran_no1_5][5]
                                arrival_data_matrix1[tran_no1_5][5] = 0
                                if arrival_data_matrix1[tran_no1_5][1] == 0 and arrival_data_matrix1[tran_no1_5][2] == 0 and arrival_data_matrix1[tran_no1_5][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix1[tran_no1_5][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index1_5 = np.where(arrival_data_matrix1[:, 5] != 0)
                                if first_data_index1_5 and len(first_data_index1_5[0] > 0):
                                    tran_no1_5 = first_data_index1_5[0][0]
                                else:
                                    print('first_data_index1_5', first_data_index1_5)
                                    tran_no1_5 = 0
                                Vars.start_trans_no1_5 = tran_no1_5  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix1[tran_no1_5][0] = arrival_data_matrix1[tran_no1_5][1] + arrival_data_matrix1[tran_no1_5][2] + arrival_data_matrix1[tran_no1_5][5]
            else:  # 没有waitpacketSize
                if action[4] == 1:
                    power_coms1 = Vars.power_idle_red * (1 - transition_time_ratio1)  # 没有等待的包，则传输功率为空闲状态减去转换的时间
                elif action[4] == 2:
                    power_coms1 = Vars.power_idle_black * (1 - transition_time_ratio1)
                elif action[4] == 3:
                    power_coms1 = Vars.power_idle_green * (1 - transition_time_ratio1)
                energy_sum += power_coms1
        elif BSmodel1 == 1:  # BS0处于SM1状态
            power_coms1 = Vars.power_SM1 * (1 - transition_time_ratio1)  # 除去转换的时间，剩余的就是睡眠时间了
            energy_sum += power_coms1  # 将能耗加到总能耗中
        elif BSmodel1 == 2:  # SM2
            power_coms1 = Vars.power_SM2 * (1 - transition_time_ratio1)
            energy_sum += power_coms1
        elif BSmodel1 == 3:  # SM3
            power_coms1 = Vars.power_SM2 * (1 - transition_time_ratio1)
            energy_sum += power_coms1
        transition_time_ratio1 = 0

    # print('BS2')
    # BS2
    if transition_time2 != 0:  # 说明当前处于work model转换阶段

        if 0 < transition_time2 < 1:  # 说明当前处于模过渡阶段，且过渡时间transition_time_ratio0小于1ms
            transition_time_ratio2 = transition_time2
            transition_time2 = 0
        elif transition_time2 > 1:  # 当前处于模式过渡阶段且过渡时间大于1ms
            transition_time2 = transition_time2 - 1
    else:
        if BSmodel2 == 0:
            # 连接了user1/3/4/5
            available_trans2_1 = trans_rate_list_user1[2] * (1 - transition_time_ratio2)  # user1
            available_trans2_3 = trans_rate_list_user3[2] * (1 - transition_time_ratio2)  # user3
            available_trans2_4 = trans_rate_list_user4[2] * (1 - transition_time_ratio2)  # user4
            available_trans2_5 = trans_rate_list_user5[2] * (1 - transition_time_ratio2)  # user5

            if wait_packetSize2 > 0:
                # for user1
                if wait_packetSize2_1 >= available_trans2_1:  # 说明使用当前全部的1ms才能刚好处理或处理不完data
                    # 先判断zooming的大小
                    if action[5] == 1:  # 当前处于red圆
                        # 计算energy
                        power_coms2_1 = Vars.power_trans_full_red * (1 - transition_time_ratio2)
                    elif action[5] == 2:  # 当前处于black圆
                        power_coms2_1 = Vars.power_trans_full_black * (1 - transition_time_ratio2)
                    elif action[5] == 3:  # 当前处于green圆
                        power_coms2_1 = Vars.power_trans_full_green * (1 - transition_time_ratio2)
                    energy_sum += power_coms2_1  # 将当前时刻处理数据消耗的功耗加到总功耗中，用于计算reward
                    trans_flag2_1 = True  # 用True标记一直传输
                    first_data_index2_1 = np.where(arrival_data_matrix2[:, 1] != 0)
                    if first_data_index2_1 and len(first_data_index2_1[0] > 0):
                        tran_no2_1 = first_data_index2_1[0][0]
                    else:
                        print('first_data_index2_1', first_data_index2_1)
                        tran_no2_1 = 0
                    Vars.start_trans_no2_1 = tran_no2_1
                    sum_trans2_1 = 0  # 指当前1ms内已经处理完的数据
                    while trans_flag2_1 == True and tran_no2_1 < len(arrival_data_matrix2):  # 还在处理数据切合法
                        if arrival_data_matrix2[tran_no2_1][1] > 0:  # user2有等待数据#指当前arrival_data_matrix2行的数据还没处理完，如第1行100，处理完50，还剩50时
                            if sum_trans2_1 + arrival_data_matrix2[tran_no2_1][1] > available_trans2_1:  # sum_trans0指当前1ms内已经消耗的处理能力，arrival_data_matrix2[trans_no][0]指当前行的剩余数据
                                # 如果当前1ms消耗的处理能力+matrix当前行的剩余数据>处理能力，说明当前的1ms处理不完arrival_data_matrix2arrival_data_matrix2[trans_no][0]的数据
                                arrival_data_matrix2[tran_no2_1][1] = arrival_data_matrix2[tran_no2_1][1] - (available_trans2_1 - sum_trans2_1)
                                trans_flag2_1 = False  # 当前1ms已经消耗完,这一秒已经过完了，直接退出这1ms，进入下1ms，下一次大循环了
                            else:  # 如果sum_trans0 + arrival_data_matrix2[trans_no][0] 小于 available_trans0，说明当前1ms内剩余的处理能力可以把当前的arrival_data_matrix2[trans_no][1]的数据处理完
                                sum_trans2_1 = sum_trans2_1 + arrival_data_matrix2[tran_no2_1][1]
                                arrival_data_matrix2[tran_no2_1][1] = 0  # 当当前行的数据被处理完
                                if arrival_data_matrix2[tran_no2_1][1] == 0 and arrival_data_matrix2[tran_no2_1][2] == 0 and arrival_data_matrix2[tran_no2_1][4] == 0 and arrival_data_matrix2[tran_no2_1][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix2[tran_no2_1][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index2_1 = np.where(arrival_data_matrix2[:, 1] != 0)
                                if first_data_index2_1 and len(first_data_index2_1[0] > 0):
                                    tran_no2_1 = first_data_index2_1[0][0]
                                else:
                                    print('first_data_index2_1', first_data_index2_1)
                                    tran_no2_1 = 0
                                Vars.start_trans_no2_1 = tran_no2_1  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix2[tran_no2_1][0] = arrival_data_matrix2[tran_no2_1][1] + \
                                                          arrival_data_matrix2[tran_no2_1][3] + \
                                                          arrival_data_matrix2[tran_no2_1][4] + \
                                                          arrival_data_matrix2[tran_no2_1][5]
                 #TODO在当前的if elif中再再加一个判断条件，如果当前的waitpacketsize为0，判断actionuser，当前user是连接的还是不连接的，由此再计算user的power
                        # 在else中没有trans_flag0设置为false（因为当前的1ms有能力处理完当前数据，而我们不确定当前1ms是否还可以继续处理数据，因此循环继续处理）
                elif (wait_packetSize2_1 > 0) and (wait_packetSize2_1 < available_trans2_1):  # 说明当前的1ms完全可以处理完当前的数据，且处理完后因为没有立刻转入SM状态，因此在activate中有一段空闲时刻
                    if action[5] == 1:
                        power_coms2_1 = (Vars.power_trans_full_red * (wait_packetSize2_1 / available_trans2_1) + Vars.power_idle_red * (1 - wait_packetSize2_1 / trans_rate_list_user1[2])) * (1 - transition_time_ratio2) / 1000  # 两部分功率：传输过程中 + 空闲状态
                    elif action[5] == 2:
                        power_coms2_1 = (Vars.power_trans_full_black * (wait_packetSize2_1 / available_trans2_1) + Vars.power_idle_black * (1 - wait_packetSize2_1 / trans_rate_list_user1[2])) * (1 - transition_time_ratio2) / 1000
                    elif action[5] == 3:
                        power_coms2_1 = (Vars.power_trans_full_green * (wait_packetSize2_1 / available_trans2_1) + Vars.power_idle_green * (1 - wait_packetSize2_1 / trans_rate_list_user1[2])) * (1 - transition_time_ratio2) / 1000
                    energy_sum += power_coms2_1  # 将功率加进来
                    trans_flag2_1 = True
                    first_data_index2_1 = np.where(arrival_data_matrix2[:, 1] != 0)
                    if first_data_index2_1 and len(first_data_index2_1[0] > 0):
                        tran_no2_1 = first_data_index2_1[0][0]
                    else:
                        print('first_data_index2_1', first_data_index2_1)
                        tran_no2_1 = 0
                    Vars.start_trans_no2_1 = tran_no2_1
                    sum_trans2_1 = 0  # 当前1ms内的已经处理的能力
                    while trans_flag2_1 == True and tran_no2_1 < len(arrival_data_matrix2) and sum_trans2_1 < wait_packetSize2_1:  # 当前时刻已经有一部分处理的数据了，但是由于大的判断条件为wait_packetSize0 < available_trans0，所以这里要判断一下sum_trans0 < wait_packetSize0，说明当前1ms内还有一部分没有处理完的waitpacketsize
                        if arrival_data_matrix2[tran_no2_1][1] > 0:
                            if sum_trans2_1 + arrival_data_matrix2[tran_no2_1][1] > available_trans2_1:
                                arrival_data_matrix2[tran_no2_1][1] = arrival_data_matrix2[tran_no2_1][1] - (available_trans2_1 - sum_trans2_1)
                                tran_no2_1 = False
                            else:
                                sum_trans2_1 = sum_trans2_1 + arrival_data_matrix2[tran_no2_1][1]
                                arrival_data_matrix2[tran_no2_1][1] = 0
                                if arrival_data_matrix2[tran_no2_1][1] == 0 and arrival_data_matrix2[tran_no2_1][3] == 0 and arrival_data_matrix2[tran_no2_1][4] == 0 and arrival_data_matrix2[tran_no2_1][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix2[tran_no2_1][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index2_1 = np.where(arrival_data_matrix2[:, 1] != 0)
                                if first_data_index2_1 and len(first_data_index2_1[0] > 0):
                                    tran_no2_1 = first_data_index2_1[0][0]
                                else:
                                    print('first_data_index2_1', first_data_index2_1)
                                    tran_no2_1 = 0
                                Vars.start_trans_no2_1 = tran_no2_1  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix2[tran_no2_1][0] = arrival_data_matrix2[tran_no2_1][1] + \
                                                          arrival_data_matrix2[tran_no2_1][3] + \
                                                          arrival_data_matrix2[tran_no2_1][4] + \
                                                          arrival_data_matrix2[tran_no2_1][5]

                # for user3
                if wait_packetSize2_3 >= available_trans2_3:  # 说明使用当前全部的1ms才能刚好处理或处理不完data
                    # 先判断zooming的大小
                    if action[5] == 1:  # 当前处于red圆
                        # 计算energy
                        power_coms2_3 = Vars.power_trans_full_red * (1 - transition_time_ratio2)
                    elif action[5] == 2:  # 当前处于black圆
                        power_coms2_3 = Vars.power_trans_full_black * (1 - transition_time_ratio2)
                    elif action[5] == 3:  # 当前处于green圆
                        power_coms2_3 = Vars.power_trans_full_green * (1 - transition_time_ratio2)
                    energy_sum += power_coms2_3  # 将当前时刻处理数据消耗的功耗加到总功耗中，用于计算reward
                    trans_flag2_3 = True  # 用True标记一直传输
                    first_data_index2_3 = np.where(arrival_data_matrix2[:, 3] != 0)
                    if first_data_index2_3 and len(first_data_index2_3[0] > 0):
                        tran_no2_3 = first_data_index2_3[0][0]
                    else:
                        print('first_data_index2_3', first_data_index2_3)
                        tran_no2_3 = 0
                    Vars.start_trans_no2_3 = tran_no2_3
                    sum_trans2_3 = 0  # 指当前1ms内已经处理完的数据
                    while trans_flag2_3 == True and tran_no2_3 < len(arrival_data_matrix2):  # 还在处理数据切合法
                        if arrival_data_matrix2[tran_no2_3][3] > 0:  # user2有等待数据#指当前arrival_data_matrix2行的数据还没处理完，如第1行100，处理完50，还剩50时
                            if sum_trans2_3 + arrival_data_matrix2[tran_no2_3][3] > available_trans2_3:  # sum_trans0指当前1ms内已经消耗的处理能力，arrival_data_matrix2[trans_no][0]指当前行的剩余数据
                                # 如果当前1ms消耗的处理能力+matrix当前行的剩余数据>处理能力，说明当前的1ms处理不完arrival_data_matrix2arrival_data_matrix2[trans_no][0]的数据
                                arrival_data_matrix2[tran_no2_3][3] = arrival_data_matrix2[tran_no2_3][3] - (available_trans2_3 - sum_trans2_3)
                                trans_flag2_3 = False  # 当前1ms已经消耗完,这一秒已经过完了，直接退出这1ms，进入下1ms，下一次大循环了
                            else:  # 如果sum_trans0 + arrival_data_matrix2[trans_no][0] 小于 available_trans0，说明当前1ms内剩余的处理能力可以把当前的arrival_data_matrix2[trans_no][3]的数据处理完
                                sum_trans2_3 = sum_trans2_3 + arrival_data_matrix2[tran_no2_3][3]
                                arrival_data_matrix2[tran_no2_3][3] = 0  # 当当前行的数据被处理完
                                if arrival_data_matrix2[tran_no2_3][1] == 0 and arrival_data_matrix2[tran_no2_3][3] == 0 and arrival_data_matrix2[tran_no2_3][4] == 0 and \
                                        arrival_data_matrix2[tran_no2_3][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix2[tran_no2_3][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index2_3 = np.where(arrival_data_matrix2[:, 3] != 0)
                                if first_data_index2_3 and len(first_data_index2_3[0] > 0):
                                    tran_no2_3 = first_data_index2_3[0][0]
                                else:
                                    print('first_data_index2_3', first_data_index2_3)
                                    tran_no2_3 = 0
                                Vars.start_trans_no2_3 = tran_no2_3  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix2[tran_no2_3][0] = arrival_data_matrix2[tran_no2_3][1] + \
                                                          arrival_data_matrix2[tran_no2_3][3] + \
                                                          arrival_data_matrix2[tran_no2_3][4] + \
                                                          arrival_data_matrix2[tran_no2_3][5]
                        # 在else中没有trans_flag0设置为false（因为当前的1ms有能力处理完当前数据，而我们不确定当前1ms是否还可以继续处理数据，因此循环继续处理）
                elif (wait_packetSize2_3 > 0) and (wait_packetSize2_3 < available_trans2_3):  # 说明当前的1ms完全可以处理完当前的数据，且处理完后因为没有立刻转入SM状态，因此在activate中有一段空闲时刻
                    if action[5] == 1:
                        power_coms2_3 = (Vars.power_trans_full_red * (wait_packetSize2_3 / available_trans2_3) + Vars.power_idle_red * (1 - wait_packetSize2_3 / trans_rate_list_user3[2])) * (1 - transition_time_ratio2) / 1000  # 两部分功率：传输过程中 + 空闲状态
                    elif action[5] == 2:
                        power_coms2_3 = (Vars.power_trans_full_black * (wait_packetSize2_3 / available_trans2_3) + Vars.power_idle_black * (1 - wait_packetSize2_3 / trans_rate_list_user3[2])) * (1 - transition_time_ratio2) / 1000
                    elif action[5] == 3:
                        power_coms2_3 = (Vars.power_trans_full_green * (wait_packetSize2_3 / available_trans2_3) + Vars.power_idle_green * (1 - wait_packetSize2_3 / trans_rate_list_user3[2])) * (1 - transition_time_ratio2) / 1000
                    energy_sum += power_coms2_3  # 将功率加进来
                    trans_flag2_3 = True
                    first_data_index2_3 = np.where(arrival_data_matrix2[:, 3] != 0)
                    if first_data_index2_3 and len(first_data_index2_3[0] > 0):
                        tran_no2_3 = first_data_index2_3[0][0]
                    else:
                        print('first_data_index2_3', first_data_index2_3)
                        tran_no2_3 = 0
                    Vars.start_trans_no2_3 = tran_no2_3
                    sum_trans2_3 = 0  # 当前1ms内的已经处理的能力
                    while trans_flag2_3 == True and tran_no2_3 < len(arrival_data_matrix2) and sum_trans2_3 < wait_packetSize2_3:  # 当前时刻已经有一部分处理的数据了，但是由于大的判断条件为wait_packetSize0 < available_trans0，所以这里要判断一下sum_trans0 < wait_packetSize0，说明当前1ms内还有一部分没有处理完的waitpacketsize
                        if arrival_data_matrix2[tran_no2_3][3] > 0:
                            if sum_trans2_3 + arrival_data_matrix2[tran_no2_3][3] > available_trans2_3:
                                arrival_data_matrix2[tran_no2_3][3] = arrival_data_matrix2[tran_no2_3][3] - (available_trans2_3 - sum_trans2_3)
                                tran_no2_3 = False
                            else:
                                sum_trans2_3 = sum_trans2_3 + arrival_data_matrix2[tran_no2_3][3]
                                arrival_data_matrix2[tran_no2_3][3] = 0
                                if arrival_data_matrix2[tran_no2_3][1] == 0 and arrival_data_matrix2[tran_no2_3][3] == 0 and arrival_data_matrix2[tran_no2_3][4] == 0 and \
                                        arrival_data_matrix2[tran_no2_3][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix2[tran_no2_3][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index2_3 = np.where(arrival_data_matrix2[:, 3] != 0)
                                if first_data_index2_3 and len(first_data_index2_3[0] > 0):
                                    tran_no2_3 = first_data_index2_3[0][0]
                                else:
                                    print('first_data_index2_3', first_data_index2_3)
                                    tran_no2_3 = 0
                                Vars.start_trans_no2_3 = tran_no2_3  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix2[tran_no2_3][0] = arrival_data_matrix2[tran_no2_3][1] + \
                                                          arrival_data_matrix2[tran_no2_3][3] + \
                                                          arrival_data_matrix2[tran_no2_3][4] + \
                                                          arrival_data_matrix2[tran_no2_3][5]

                # for user4
                if wait_packetSize2_4 >= available_trans2_4:  # 说明使用当前全部的1ms才能刚好处理或处理不完data
                    # 先判断zooming的大小
                    if action[5] == 1:  # 当前处于red圆
                        # 计算energy
                        power_coms2_4 = Vars.power_trans_full_red * (1 - transition_time_ratio2)
                    elif action[5] == 2:  # 当前处于black圆
                        power_coms2_4 = Vars.power_trans_full_black * (1 - transition_time_ratio2)
                    elif action[5] == 3:  # 当前处于green圆
                        power_coms2_4 = Vars.power_trans_full_green * (1 - transition_time_ratio2)
                    energy_sum += power_coms2_4  # 将当前时刻处理数据消耗的功耗加到总功耗中，用于计算reward
                    trans_flag2_4 = True  # 用True标记一直传输
                    first_data_index2_4 = np.where(arrival_data_matrix2[:, 4] != 0)
                    if first_data_index2_4 and len(first_data_index2_4[0] > 0):
                        tran_no2_4 = first_data_index2_4[0][0]
                    else:
                        print('first_data_index2_4', first_data_index2_4)
                        tran_no2_4 = 0
                    Vars.start_trans_no2_4 = tran_no2_4
                    sum_trans2_4 = 0  # 指当前1ms内已经处理完的数据
                    while trans_flag2_4 == True and tran_no2_4 < len(arrival_data_matrix2):  # 还在处理数据切合法
                        if arrival_data_matrix2[tran_no2_4][4] > 0:  # user2有等待数据#指当前arrival_data_matrix2行的数据还没处理完，如第1行100，处理完50，还剩50时
                            if sum_trans2_4 + arrival_data_matrix2[tran_no2_4][4] > available_trans2_4:  # sum_trans0指当前1ms内已经消耗的处理能力，arrival_data_matrix2[trans_no][0]指当前行的剩余数据
                                # 如果当前1ms消耗的处理能力+matrix当前行的剩余数据>处理能力，说明当前的1ms处理不完arrival_data_matrix2arrival_data_matrix2[trans_no][0]的数据
                                arrival_data_matrix2[tran_no2_4][4] = arrival_data_matrix2[tran_no2_4][4] - (available_trans2_4 - sum_trans2_4)
                                trans_flag2_4 = False  # 当前1ms已经消耗完,这一秒已经过完了，直接退出这1ms，进入下1ms，下一次大循环了
                            else:  # 如果sum_trans0 + arrival_data_matrix2[trans_no][0] 小于 available_trans0，说明当前1ms内剩余的处理能力可以把当前的arrival_data_matrix2[trans_no][4]的数据处理完
                                sum_trans2_4 = sum_trans2_4 + arrival_data_matrix2[tran_no2_4][4]
                                arrival_data_matrix2[tran_no2_4][4] = 0  # 当当前行的数据被处理完
                                if arrival_data_matrix2[tran_no2_4][1] == 0 and arrival_data_matrix2[tran_no2_4][3] == 0 and arrival_data_matrix2[tran_no2_4][4] == 0 and \
                                        arrival_data_matrix2[tran_no2_4][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix2[tran_no2_4][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index2_4 = np.where(arrival_data_matrix2[:, 4] != 0)
                                if first_data_index2_4 and len(first_data_index2_4[0] > 0):
                                    tran_no2_4 = first_data_index2_4[0][0]
                                else:
                                    print('first_data_index2_4', first_data_index2_4)
                                    tran_no2_4 = 0
                                Vars.start_trans_no2_4 = tran_no2_4  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix2[tran_no2_4][0] = arrival_data_matrix2[tran_no2_4][1] + \
                                                          arrival_data_matrix2[tran_no2_4][3] + \
                                                          arrival_data_matrix2[tran_no2_4][4] + \
                                                          arrival_data_matrix2[tran_no2_4][5]
                        # 在else中没有trans_flag0设置为false（因为当前的1ms有能力处理完当前数据，而我们不确定当前1ms是否还可以继续处理数据，因此循环继续处理）
                elif (wait_packetSize2_4 > 0) and (wait_packetSize2_4 < available_trans2_4):  # 说明当前的1ms完全可以处理完当前的数据，且处理完后因为没有立刻转入SM状态，因此在activate中有一段空闲时刻
                    if action[5] == 1:
                        power_coms2_4 = (Vars.power_trans_full_red * (wait_packetSize2_4 / available_trans2_4) + Vars.power_idle_red * (1 - wait_packetSize2_4 / trans_rate_list_user4[2])) * (1 - transition_time_ratio2) / 1000  # 两部分功率：传输过程中 + 空闲状态
                    elif action[5] == 2:
                        power_coms2_4 = (Vars.power_trans_full_black * (wait_packetSize2_4 / available_trans2_4) + Vars.power_idle_black * (1 - wait_packetSize2_4 / trans_rate_list_user4[2])) * (1 - transition_time_ratio2) / 1000
                    elif action[5] == 3:
                        power_coms2_4 = (Vars.power_trans_full_green * (wait_packetSize2_4 / available_trans2_4) + Vars.power_idle_green * (1 - wait_packetSize2_4 / trans_rate_list_user4[2])) * (1 - transition_time_ratio2) / 1000
                    energy_sum += power_coms2_4  # 将功率加进来
                    trans_flag2_4 = True
                    first_data_index2_4 = np.where(arrival_data_matrix2[:, 4] != 0)
                    if first_data_index2_4 and len(first_data_index2_4[0] > 0):
                        tran_no2_4 = first_data_index2_4[0][0]
                    else:
                        print('first_data_index2_4', first_data_index2_4)
                        tran_no2_4 = 0
                    Vars.start_trans_no2_4 = tran_no2_4
                    sum_trans2_4 = 0  # 当前1ms内的已经处理的能力
                    while trans_flag2_4 == True and tran_no2_4 < len(arrival_data_matrix2) and sum_trans2_4 < wait_packetSize2_4:  # 当前时刻已经有一部分处理的数据了，但是由于大的判断条件为wait_packetSize0 < available_trans0，所以这里要判断一下sum_trans0 < wait_packetSize0，说明当前1ms内还有一部分没有处理完的waitpacketsize
                        if arrival_data_matrix2[tran_no2_4][4] > 0:
                            if sum_trans2_4 + arrival_data_matrix2[tran_no2_4][4] > available_trans2_4:
                                arrival_data_matrix2[tran_no2_4][4] = arrival_data_matrix2[tran_no2_4][4] - (available_trans2_4 - sum_trans2_4)
                                tran_no2_4 = False
                            else:
                                sum_trans2_4 = sum_trans2_4 + arrival_data_matrix2[tran_no2_4][4]
                                arrival_data_matrix2[tran_no2_4][4] = 0
                                if arrival_data_matrix2[tran_no2_4][1] == 0 and arrival_data_matrix2[tran_no2_4][3] == 0 and arrival_data_matrix2[tran_no2_4][4] == 0 and \
                                        arrival_data_matrix2[tran_no2_4][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix2[tran_no2_4][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index2_4 = np.where(arrival_data_matrix2[:, 4] != 0)
                                if first_data_index2_4 and len(first_data_index2_4[0] > 0):
                                    tran_no2_4 = first_data_index2_4[0][0]
                                else:
                                    print('first_data_index2_4', first_data_index2_4)
                                    tran_no2_4 = 0
                                Vars.start_trans_no2_4 = tran_no2_4  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix2[tran_no2_4][0] = arrival_data_matrix2[tran_no2_4][1] + \
                                                          arrival_data_matrix2[tran_no2_4][3] + \
                                                          arrival_data_matrix2[tran_no2_4][4] + \
                                                          arrival_data_matrix2[tran_no2_4][5]

                # for user5
                if wait_packetSize2_5 >= available_trans2_5:  # 说明使用当前全部的1ms才能刚好处理或处理不完data
                    # 先判断zooming的大小
                    if action[5] == 1:  # 当前处于red圆
                        # 计算energy
                        power_coms2_5 = Vars.power_trans_full_red * (1 - transition_time_ratio2)
                    elif action[5] == 2:  # 当前处于black圆
                        power_coms2_5 = Vars.power_trans_full_black * (1 - transition_time_ratio2)
                    elif action[5] == 3:  # 当前处于green圆
                        power_coms2_5 = Vars.power_trans_full_green * (1 - transition_time_ratio2)
                    energy_sum += power_coms2_5  # 将当前时刻处理数据消耗的功耗加到总功耗中，用于计算reward
                    trans_flag2_5 = True  # 用True标记一直传输
                    first_data_index2_5 = np.where(arrival_data_matrix2[:, 5] != 0)
                    if first_data_index2_5 and len(first_data_index2_5[0] > 0):
                        tran_no2_5 = first_data_index2_5[0][0]
                    else:
                        print('first_data_index2_5', first_data_index2_5)
                        tran_no2_5 = 0

                    Vars.start_trans_no2_5 = tran_no2_5
                    sum_trans2_5 = 0  # 指当前1ms内已经处理完的数据
                    while trans_flag2_5 == True and tran_no2_5 < len(arrival_data_matrix2):  # 还在处理数据切合法
                        if arrival_data_matrix2[tran_no2_5][5] > 0:  # user2有等待数据#指当前arrival_data_matrix2行的数据还没处理完，如第1行100，处理完50，还剩50时
                            if sum_trans2_5 + arrival_data_matrix2[tran_no2_5][5] > available_trans2_5:  # sum_trans0指当前1ms内已经消耗的处理能力，arrival_data_matrix2[trans_no][0]指当前行的剩余数据
                                # 如果当前1ms消耗的处理能力+matrix当前行的剩余数据>处理能力，说明当前的1ms处理不完arrival_data_matrix2arrival_data_matrix2[trans_no][0]的数据
                                arrival_data_matrix2[tran_no2_5][5] = arrival_data_matrix2[tran_no2_5][5] - (available_trans2_5 - sum_trans2_5)
                                trans_flag2_5 = False  # 当前1ms已经消耗完,这一秒已经过完了，直接退出这1ms，进入下1ms，下一次大循环了
                            else:  # 如果sum_trans0 + arrival_data_matrix2[trans_no][0] 小于 available_trans0，说明当前1ms内剩余的处理能力可以把当前的arrival_data_matrix2[trans_no][5]的数据处理完
                                sum_trans2_5 = sum_trans2_5 + arrival_data_matrix2[tran_no2_5][5]
                                arrival_data_matrix2[tran_no2_5][5] = 0  # 当当前行的数据被处理完
                                if arrival_data_matrix2[tran_no2_5][1] == 0 and arrival_data_matrix2[tran_no2_5][3] == 0 and arrival_data_matrix2[tran_no2_5][4] == 0 and \
                                        arrival_data_matrix2[tran_no2_5][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix2[tran_no2_5][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index2_5 = np.where(arrival_data_matrix2[:, 5] != 0)
                                if first_data_index2_5 and len(first_data_index2_5[0] > 0):
                                    tran_no2_5 = first_data_index2_5[0][0]
                                else:
                                    print('first_data_index2_5', first_data_index2_5)
                                    tran_no2_5 = 0
                                Vars.start_trans_no2_5 = tran_no2_5  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix2[tran_no2_5][0] = arrival_data_matrix2[tran_no2_5][1] + \
                                                          arrival_data_matrix2[tran_no2_5][3] + \
                                                          arrival_data_matrix2[tran_no2_5][4] + \
                                                          arrival_data_matrix2[tran_no2_5][5]
                        # 在else中没有trans_flag0设置为false（因为当前的1ms有能力处理完当前数据，而我们不确定当前1ms是否还可以继续处理数据，因此循环继续处理）
                elif (wait_packetSize2_5 > 0) and (wait_packetSize2_5 < available_trans2_5):  # 说明当前的1ms完全可以处理完当前的数据，且处理完后因为没有立刻转入SM状态，因此在activate中有一段空闲时刻
                    if action[5] == 1:
                        power_coms2_5 = (Vars.power_trans_full_red * (wait_packetSize2_5 / available_trans2_5) + Vars.power_idle_red * (1 - wait_packetSize2_5 / trans_rate_list_user5[2])) * (1 - transition_time_ratio2) / 1000  # 两部分功率：传输过程中 + 空闲状态
                    elif action[5] == 2:
                        power_coms2_5 = (Vars.power_trans_full_black * (wait_packetSize2_5 / available_trans2_5) + Vars.power_idle_black * (1 - wait_packetSize2_5 / trans_rate_list_user5[2])) * (1 - transition_time_ratio2) / 1000
                    elif action[5] == 3:
                        power_coms2_5 = (Vars.power_trans_full_green * (wait_packetSize2_5 / available_trans2_5) + Vars.power_idle_green * (1 - wait_packetSize2_5 / trans_rate_list_user5[2])) * (1 - transition_time_ratio2) / 1000
                    energy_sum += power_coms2_5  # 将功率加进来
                    trans_flag2_5 = True
                    first_data_index2_5 = np.where(arrival_data_matrix2[:, 5] != 0)
                    if first_data_index2_5 and len(first_data_index2_5[0] > 0):
                        tran_no2_5 = first_data_index2_5[0][0]
                    else:
                        print('first_data_index2_5', first_data_index2_5)
                        tran_no2_5 = 0
                    Vars.start_trans_no2_5 = tran_no2_5
                    sum_trans2_5 = 0  # 当前1ms内的已经处理的能力
                    while trans_flag2_5 == True and tran_no2_5 < len(arrival_data_matrix2) and sum_trans2_5 < wait_packetSize2_5:  # 当前时刻已经有一部分处理的数据了，但是由于大的判断条件为wait_packetSize0 < available_trans0，所以这里要判断一下sum_trans0 < wait_packetSize0，说明当前1ms内还有一部分没有处理完的waitpacketsize
                        if arrival_data_matrix2[tran_no2_5][5] > 0:
                            if sum_trans2_5 + arrival_data_matrix2[tran_no2_5][5] > available_trans2_5:
                                arrival_data_matrix2[tran_no2_5][5] = arrival_data_matrix2[tran_no2_5][5] - (available_trans2_5 - sum_trans2_5)
                                tran_no2_5 = False
                            else:
                                sum_trans2_5 = sum_trans2_5 + arrival_data_matrix2[tran_no2_5][5]
                                arrival_data_matrix2[tran_no2_5][5] = 0
                                if arrival_data_matrix2[tran_no2_5][1] == 0 and arrival_data_matrix2[tran_no2_5][3] == 0 and arrival_data_matrix2[tran_no2_5][5] == 0 and \
                                        arrival_data_matrix2[tran_no2_5][5] == 0:
                                    # 当user2，3，5的当前行数据都为0时，才说明当前行的数据已经被处理完，所以此时才能记录完成时间，否则不记录
                                    arrival_data_matrix2[tran_no2_5][7] = Vars.time  # 记录处理完成数据的时刻，用于计算latency
                                else:
                                    pass
                                first_data_index2_5 = np.where(arrival_data_matrix2[:, 5] != 0)
                                if first_data_index2_5 and len(first_data_index2_5[0] > 0):
                                    tran_no2_5 = first_data_index2_5[0][0]
                                else:
                                    print('first_data_index2_5', first_data_index2_5)
                                    tran_no2_5 = 0
                                Vars.start_trans_no2_5 = tran_no2_5  # 转入下一次有数据的行进行数据处理#转入下一行处理数据
                    # TODO 更新arrival_data_matrix的第0列
                    arrival_data_matrix2[tran_no2_5][0] = arrival_data_matrix2[tran_no2_5][1] + \
                                                          arrival_data_matrix2[tran_no2_5][3] + \
                                                          arrival_data_matrix2[tran_no2_5][4] + \
                                                          arrival_data_matrix2[tran_no2_5][5]

            else:  # 没有waitpacketSize
                if action[5] == 1:
                    power_coms2 = Vars.power_idle_red * (1 - transition_time_ratio2)  # 没有等待的包，则传输功率为空闲状态减去转换的时间
                elif action[5] == 2:
                    power_coms2 = Vars.power_idle_black * (1 - transition_time_ratio2)
                elif action[5] == 3:
                    power_coms2 = Vars.power_idle_green * (1 - transition_time_ratio2)
                energy_sum += power_coms2
        elif BSmodel2 == 1:  # BS0处于SM1状态
            power_coms2 = Vars.power_SM1 * (1 - transition_time_ratio2)  # 除去转换的时间，剩余的就是睡眠时间了
            energy_sum += power_coms2  # 将能耗加到总能耗中
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
#注意：这个transition函数的作用是计算在转换过程中的转换功率，而不是定义BS是怎么转换的，因为BSmodel可以直接根据action进行转换，不需要知道上一个model是什么，需要知道上一个model是什么的只有计算转换过程需要
def transition(observation,action,energy_sum,transition_time_ratio0,transition_time_ratio1,transition_time_ratio2,transition_time0,transition_time1,transition_time2,mode_time0,mode_time1,mode_time2):#注意：这里的BSmodel0,BSmodel1,BSmodel2是指的当前时刻的state中的元素
    #TODO 处理时，处理的是arrival_data的各个user的，就不合起来了，但是需要index0的data家和做一个waitpacketsize，因此，在函数最后，要更新一下arrival-data的inedx0
    #首先判断当前时刻BS work model是什么，再根据action将BS work model转换为新的work model,所以这里的BSmode0/1/2也是state中的，而不是action中的
    BSmodel = observation['BSmodel']
    BSmodel0 = BSmodel[0]
    BSmodel1 = BSmodel[1]
    BSmodel2 = BSmodel[2]
    #BS0
    if transition_time0 != 0:
        pass
    else:
        #TODO 这里有个问题，是需要从正常的功率到SM吗，还是直接从zooming到转换时间
        if BSmodel0 == 0:#Avtivate
            if action[0] == 1: #activate-SM1
                BSmodel0 = 1
                transition_time_ratio0 = Vars.delay_Active_SM1 #转换时间为0.0355
                # if action[3] == 1:
                #     power_coms_trans0 = Vars.power_idle_red * transition_time_ratio0#计算转换时的功率
                # elif action[3] ==2:
                power_coms_trans0 = Vars.power_idle_black * transition_time_ratio0
                # elif action[3] ==3:
                #     power_coms_trans0 = Vars.power_idle_green * transition_time_ratio0
                energy_sum += power_coms_trans0
                transition_time0 = 0 #转换完后，将转换时间重置为0
                mode_time0 = 0#指model的持续时间
            elif action[0] == 2: #activate-SM2,则要先经过SM1
                BSmodel0 = 2
                transition_time_ratio0 = Vars.delay_Active_SM1 + Vars.delay_SM1_SM2 #0.0355+0.5
                # if action[3] == 1:
                #     power_coms_trans0 = Vars.power_idle_red * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 #activate功率*activate—SM1的延迟时间 + SM1的时间*SM1_SM2的延迟  注：既然是延迟，就是如SM1-SM2，就是在SM1停留延迟时间后，再转换到SM2
                # elif action[3] == 2:
                power_coms_trans0 = Vars.power_idle_black * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2
                # elif action[3] == 3:
                #     power_coms_trans0 = Vars.power_idle_green * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2
                energy_sum += power_coms_trans0
                SM_Hold_Time0 = Vars.hold_time_SM2
                transition_time0 = SM_Hold_Time0-(1-transition_time_ratio0)
                energy_sum += Vars.power_SM2 * Vars.hold_time_SM2 #SM2的hole功率
                mode_time0 = 0
            elif action[0] == 3:  # active-SM3
                BSmodel0 = 3
                # if action[3] == 1:
                #     power_coms_transition0 = Vars.power_idle_red * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                # elif action[3] == 2:
                power_coms_transition0 = Vars.power_idle_black * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                # elif action[3] == 3:
                #     power_coms_transition0 = Vars.power_idle_green * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
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
                transition_time0 = Vars.delay_SM1_SM2 + Vars.delay_SM2_SM3
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
                # if action[4] == 1:
                #     power_coms_trans1 = Vars.power_idle_red * transition_time_ratio1#计算转换时的功率
                # elif action[4] == 2:
                power_coms_trans1 = Vars.power_idle_black * transition_time_ratio1
                # elif action[4] == 3:
                #     power_coms_trans1 = Vars.power_idle_green * transition_time_ratio1
                energy_sum += power_coms_trans1
                transition_time1 = 0 #转换完后，将转换时间重置为0
                mode_time1 = 0#指model的持续时间
            elif action[1] == 2: #activate-SM2,则要先经过SM1
                BSmodel1 = 2
                transition_time_ratio1 = Vars.delay_Active_SM1 + Vars.delay_SM1_SM2 #0.0355+0.5
                # if action[4] == 1:
                #     power_coms_trans1 = Vars.power_idle_red * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 #activate功率*activate—SM1的延迟时间 + SM1的时间*SM1_SM2的延迟  注：既然是延迟，就是如SM1-SM2，就是在SM1停留延迟时间后，再转换到SM2
                # elif action[4] == 2:
                power_coms_trans1 = Vars.power_idle_black * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2
                # if action[4] == 3:
                #     power_coms_trans1 = Vars.power_idle_green * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2
                energy_sum += power_coms_trans1
                SM_Hold_Time1 = Vars.hold_time_SM2
                transition_time1 = SM_Hold_Time1 - (1-transition_time_ratio1)
                energy_sum += Vars.power_SM2 * Vars.hold_time_SM2 #SM2的hole功率
                mode_time1 = 0
            elif action[1] == 3:  # active-SM3 注意：转换成了SMmolde后，BS的zooming大小就变成0了，因此不需要分类zooming大小是啥了
                BSmodel1 = 3
                # if action[4] == 1:
                #     power_coms_transition1 = Vars.power_idle_red * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                # elif action[4] == 2:
                power_coms_transition1 = Vars.power_idle_black * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                # elif action[4] == 3:
                #     power_coms_transition1 = Vars.power_idle_green * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
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
                transition_time1 = Vars.delay_SM1_SM2 + Vars.delay_SM2_SM3
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
                # if action[5] == 1:
                #     power_coms_trans2 = Vars.power_idle_red * transition_time_ratio2#计算转换时的功率
                # elif action[5] == 2:
                power_coms_trans2 = Vars.power_idle_black * transition_time_ratio2
                # elif action[5] == 3:
                #     power_coms_trans2 = Vars.power_idle_green * transition_time_ratio2
                energy_sum += power_coms_trans2
                transition_time2 = 0 #转换完后，将转换时间重置为0
                mode_time2 = 0#指model的持续时间
            elif action[2] == 2: #activate-SM2,则要先经过SM1
                BSmodel2 = 2
                transition_time_ratio2 = Vars.delay_Active_SM1 + Vars.delay_SM1_SM2 #0.0355+0.5
                # if action[5] == 1:
                #     power_coms_trans2 = Vars.power_idle_red * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 #activate功率*activate—SM1的延迟时间 + SM1的时间*SM1_SM2的延迟  注：既然是延迟，就是如SM1-SM2，就是在SM1停留延迟时间后，再转换到SM2
                # elif action[5] == 2:
                power_coms_trans2 = Vars.power_idle_black * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2
                # elif action[5] == 3:
                #     power_coms_trans2 = Vars.power_idle_green * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2
                energy_sum += power_coms_trans2
                SM_Hold_Time2 = Vars.hold_time_SM2
                transition_time2 = SM_Hold_Time2 - (1-transition_time_ratio2)
                energy_sum += Vars.power_SM2 * Vars.hold_time_SM2 #SM2的hole功率
                mode_time2 = 0
            elif action[2] == 3:  # active-SM3
                BSmodel2 = 3
                # if action[5] == 1:
                #     power_coms_transition2 = Vars.power_idle_red * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                # elif action[5] == 2:
                power_coms_transition2 = Vars.power_idle_black * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
                # elif action[5] == 3:
                #     power_coms_transition2 = Vars.power_idle_green * Vars.delay_Active_SM1 + Vars.power_SM1 * Vars.delay_SM1_SM2 + Vars.power_SM2 * Vars.delay_SM2_SM3
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
                transition_time2 = Vars.delay_SM1_SM2 + Vars.delay_SM2_SM3
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
    return energy_sum,transition_time_ratio0,transition_time_ratio1,transition_time_ratio2,transition_time0,transition_time1,transition_time2,mode_time0,mode_time1,mode_time2
#return BSmodel0,BSmodel1,BSmodel2,energy_sum,transition_time_ratio0,transition_time_ratio1,transition_time_ratio2,transition_time0,transition_time1,transition_time2,mode_time0,mode_time1,mode_time2


#先定义一个LSTM的function

def LSTM_Train(DataSet,scaler):
    # 读取数据
    data = pd.read_csv(DataSet)
    # 将字符串数据转换为列表
    data_list = list(map(int, data.iloc[:, 0]))
    # data_list = list(map(int, data.split()))
    # 转换为NumPy数组
    data_array = np.array(data_list)

    # 将数据归一化到0-1范围
    scaled_data = scaler.fit_transform(data_array.reshape(-1, 1))
    # 创建训练数据
    #look_back = 5  # 设置滑动窗口的大小
    X, y = [], []

    for i in range(len(scaled_data) - Vars.look_back):
        # 每循环一次，取一次样本
        X.append(scaled_data[i:(i + Vars.look_back)].flatten())  # 每次在 X 中添加滑动窗口个数据（0-5秒的数据）
        y.append(scaled_data[i + Vars.look_back])  # 在 y 中添加 label，即下个时刻的数值，用作 y （第6秒的数据）

    # 手动切分数据集
    # 划分数据集和训练集的比例
    train_size = 0.8
    #test_size = 1 - train_size

    # 计算切割点
    split_index = int(len(X) * train_size)

    # 切割数据
    X_train, y_train = np.array(X[:split_index]), np.array(y[:split_index])
    X_test, y_test = np.array(X[split_index:]), np.array(y[split_index:])

    # 将数据转换为适合 LSTM 输入的形状
    X_train = X_train.reshape((X_train.shape[0], Vars.look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], Vars.look_back, 1))

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(Vars.look_back, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)
    # 在测试集上评估模型
    predicted_values = model.predict(X_test)

    # 将预测值反归一化

    predicted_values = scaler.inverse_transform(predicted_values)
    y_test = scaler.inverse_transform(y_test)

    # 打印前几个预测值和实际值
    # for i in range(5):
        # print(f'Predicted: {predicted_values[i][0]}, Actual: {y_test[i][0]}')
    user1_model = model
    return user1_model, scaler



def LSTM_predict(model,Dataset,scaler,counter_for_LSTM):
    # 主循环
    input_data = pd.read_csv(Dataset)
    input_data = list(map(int, input_data.iloc[:, 0]))
    input_data = np.array(input_data)
    #在这里加上了，如果最后5个值都为0了，则，预测值为一个很大的值吧，否则用LSTM的预测值
    last_five_data = input_data[0+counter_for_LSTM:Vars.look_back+counter_for_LSTM]
    last_ele_zero = np.all(last_five_data[4] == 0)
    if last_ele_zero:#如果最后一个的数据为0，说明已经到最后了
        predicted_value_user1 = 3001
    else:#否则，用LSTM预测
        last_data = input_data[0+counter_for_LSTM:Vars.look_back+counter_for_LSTM].reshape((1, Vars.look_back, 1))  # 获取测试集中最后一个样本的滑动窗口数据
        # 使用模型进行预测
        predicted_value = model.predict(last_data)
        # 反归一化预测值
        predicted_value_final = scaler.inverse_transform(predicted_value)
        predicted_value_user1 = round(predicted_value_final[0][0])
    # 将predicted_value_final的值四舍五入
    # 打印未来的预测值
    # predicted_value_final_list.append(predicted_value_final)

    # print(f'Future Prediction: {predicted_value_user1}')
    return predicted_value_user1

def action_choice(action_space_index_total,state_space, action_space_matrix): #根据state选择action
    #1. for waitpacketSize的选择
    waitpacketSize = state_space[0]
    action_space_matrix = action_space_matrix
    waitpacketSize0 = waitpacketSize[0] + waitpacketSize[1]+waitpacketSize[2]+waitpacketSize[3]+waitpacketSize[4]
    waitpacketSize1 = waitpacketSize[5] + waitpacketSize[6]+waitpacketSize[7]+waitpacketSize[8]+waitpacketSize[9]
    waitpacketSize2 = waitpacketSize[10] + waitpacketSize[11]+waitpacketSize[12]+waitpacketSize[13]+waitpacketSize[14]

    #BS0
    if waitpacketSize0 != 0: #去掉当前BS0有SM的action
        #选择BS为SMmodel的action的index
        action_illegal_for_BS0 = np.nonzero(action_space_matrix[:,0])[0]
        action_space_index_legal_for_BS0 = [num for num in action_space_index_total if num not in action_illegal_for_BS0]
    else:
        action_space_index_legal_for_BS0 = action_space_index_total
    #BS1
    if waitpacketSize1 !=0:
        action_illegal_for_BS1 = np.nonzero(action_space_matrix[:,1])[0]
        action_space_index_legal_for_BS1 = [num for num in action_space_index_legal_for_BS0 if num not in action_illegal_for_BS1]
    else:
        action_space_index_legal_for_BS1 = action_space_index_legal_for_BS0
    if waitpacketSize2 != 0:
        action_illegal_for_BS2 = np.nonzero(action_space_matrix[:,2])[0]
        action_space_index_legal_for_BS2 = [num for num in action_space_index_legal_for_BS1 if num not in action_illegal_for_BS2]
    else:
        action_space_index_legal_for_BS2 = action_space_index_legal_for_BS1
    '''
    userdata的排除是继上边waitpacket排出后的再排除
    '''
    #2. for user data的选择
    #for user1
    #如果当前时间 == LSTM预测的到达时间，则说明user有数据达到，否则删除连接的action
    #或者判断，当前的LSTM值为0还是1，0表示下一时刻user没有数据，1表示下一时刻user有数据
    LSTM = state_space[2]
    LSTM_user1 = LSTM[5]
    LSTM_user2 = LSTM[6]
    LSTM_user3 = LSTM[7]
    LSTM_user4 = LSTM[8]
    LSTM_user5 = LSTM[9]
    #for user1 index6
    #for BS0 的用户连接 只需要判断user2/3/5就可以了，因此其他情况已经在action——space中去掉了
    #user2 action inde7/waitpacketSize index1
    #for BS1 连接user1/2/5
    #user1 action index 6/waitpacketSize index 5/10
    if LSTM_user1 == 0:#连接，删掉不连接的情况
        if waitpacketSize[5] != 0 or waitpacketSize[10] !=0:
            if waitpacketSize[5] != 0:#BS1#虽然下一时刻user2没有数据进来，因为LSTM为0，但是user2存在pending，因此需要继续连接
                action_illegal_for_user1 = np.where(action_space_matrix[:, 6] != 1)[0]
            if waitpacketSize[10] !=0:#BS2
                action_illegal_for_user1 = np.where(action_space_matrix[:, 6] != 2)[0]
        else:#不连接，删掉连接的情况
            action_illegal_for_user1 = np.where(action_space_matrix[:, 6] != -1)[0]
    else:
        action_illegal_for_user1 = np.where(action_space_matrix[:, 6] == -1)[0]
    action_space_index_legal_for_user1 = [num for num in action_space_index_legal_for_BS2 if num not in action_illegal_for_user1]
    #user2  action inde7/waitpacketSize index1/6
    if LSTM_user2 == 0:
        if waitpacketSize[1] != 0 or waitpacketSize[6] != 0:  # 虽然下一时刻user2没有数据进来，因为LSTM为0，但是user2存在pending，因此需要继续连接
            if waitpacketSize[1] != 0:#BS0
                action_illegal_for_user2 = np.where(action_space_matrix[:, 7] != 0)[0]
            if waitpacketSize[6] != 0:#BS1
                action_illegal_for_user2 = np.where(action_space_matrix[:, 7] != 1)[0]
        else:
            action_illegal_for_user2 = np.where(action_space_matrix[:, 7] != -1)[0]
    else:  # 无论有没有wait，下一时刻user2有数据进来，因此，必须连接
        action_illegal_for_user2 = np.where(action_space_matrix[:, 7] == -1)[0]
    action_space_index_legal_for_user2 = [num for num in action_space_index_legal_for_user1 if num not in action_illegal_for_user2]

    #user3 action index 8/waitpacketSize index2/12
    if LSTM_user3 == 0:
        if waitpacketSize[2] != 0 or waitpacketSize[12] !=0:
            if waitpacketSize[2] != 0:#BS0
                action_illegal_for_user3 = np.where(action_space_matrix[:, 8] != 0)[0]
            if waitpacketSize[12] !=0:#BS2
                action_illegal_for_user3 = np.where(action_space_matrix[:, 8] != 2)[0]
        else:
            action_illegal_for_user3 = np.where(action_space_matrix[:, 8] != -1)[0]
    else:
        action_illegal_for_user3 = np.where(action_space_matrix[:, 8] == -1)[0]
    action_space_index_legal_for_user3 = [num for num in action_space_index_legal_for_user2 if num not in action_illegal_for_user3]
    #user4 action index 9/waitpacketSize index 13
    if LSTM_user4 == 0:
        if waitpacketSize[13] != 0:
            action_illegal_for_user4 = np.where(action_space_matrix[:, 9] ==-1)[0]
        else:
            action_illegal_for_user4 = np.where(action_space_matrix[:, 9] != -1)[0]

    else:
        action_illegal_for_user4 = np.where(action_space_matrix[:, 9] == -1)[0]
    action_space_index_legal_for_user4 = [num for num in action_space_index_legal_for_user3 if num not in action_illegal_for_user4]
    #user5 action index 10/waitpacketSize index 4/9/14
    if LSTM_user5 ==0:
        if waitpacketSize[4] != 0 or waitpacketSize[9] != 0 or waitpacketSize[14] != 0:
            if waitpacketSize[4] != 0:#BS0
                action_illegal_for_user5 = np.where(action_space_matrix[:, 10] != 0)[0]
            if waitpacketSize[9] != 0:#BS1
                action_illegal_for_user5 = np.where(action_space_matrix[:, 10] != 1)[0]
            if waitpacketSize[14] != 0:#BS3
                action_illegal_for_user5 = np.where(action_space_matrix[:, 10] != 2)[0]
        else:
            action_illegal_for_user5 = np.where(action_space_matrix[:, 10] != -1)[0]
    else:
        action_illegal_for_user5 = np.where(action_space_matrix[:, 10] == -1)[0]
    action_space_index_legal_for_user5 = [num for num in action_space_index_legal_for_user4 if num not in action_illegal_for_user5]

    action_space_legal = np.array(action_space_index_legal_for_user5)
    check_matrix = action_space_matrix[action_space_legal, :]
    return action_space_legal, check_matrix



