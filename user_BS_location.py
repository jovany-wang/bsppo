
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# BS坐标
BS0_loc_draw = (1300, 1616)
BS1_loc_draw = (800, 750)
BS2_loc_draw = (1800, 750)

# 用户坐标
user1_loc_draw = (1065,750)#BS1，BS2
user2_loc_draw = (1150,1400)#BS0，BS1
user3_loc_draw = (1700,1200)#BS0，BS2
user4_loc_draw = (1700,600)#
user5_loc_draw = (1300,1050)#BS0，BS1，BS2

# 创建一个新的绘图
fig, ax = plt.subplots()

# 绘制BS坐标
#for BS
global r_init #基站覆盖半径
r_init= 500
r_fast = r_init
plt.scatter(*BS0_loc_draw, color='red', label='BS0', marker='o')
plt.scatter(*BS1_loc_draw, color='red', label='BS1', marker='o')
plt.scatter(*BS2_loc_draw, color='red', label='BS2', marker='o')

# 绘制用户坐标
plt.scatter(*user1_loc_draw, color='black', label='User1', marker='o')
plt.scatter(*user2_loc_draw, color='black', label='User2', marker='o')
plt.scatter(*user3_loc_draw, color='black', label='User3', marker='o')
plt.scatter(*user4_loc_draw, color='black', label='User4', marker='o')
plt.scatter(*user5_loc_draw, color='black', label='User5', marker='o')

#绘制圆
circle_BS0 = patches.Circle(BS0_loc_draw, radius=r_init, fill=False, color='black', linestyle='solid')
circle_BS1 = patches.Circle(BS1_loc_draw, radius=r_init, fill=False, color='black', linestyle='solid')
circle_BS2 = patches.Circle(BS2_loc_draw, radius=r_init, fill=False, color='black', linestyle='solid')

#BS0 Zooming 圆大小
circle_BS0_in = patches.Circle(BS0_loc_draw, radius=300, fill=False, color='red', linestyle='dashed')
circle_BS0_out = patches.Circle(BS0_loc_draw, radius=750, fill=False, color='green', linestyle='dashed')

#BS1 Zooming 圆大小
circle_BS1_in = patches.Circle(BS1_loc_draw, radius=300, fill=False, color='red', linestyle='dashed')
circle_BS1_out = patches.Circle(BS1_loc_draw, radius=750, fill=False, color='green', linestyle='dashed')

#BS2 Zooming 圆大小
circle_BS2_in = patches.Circle(BS2_loc_draw, radius=300, fill=False, color='red', linestyle='dashed')
circle_BS2_out = patches.Circle(BS2_loc_draw, radius=750, fill=False, color='green', linestyle='dashed')


# circle_BS0_change = patches.Circle(BS0_loc_draw, radius=700, fill=False, color='red', linestyle='dashed')
# circle_BS1_change = patches.Circle(BS1_loc_draw, radius=300, fill=False, color='red', linestyle='dashed')
# circle_BS2_change = patches.Circle(BS2_loc_draw, radius=200, fill=False, color='red', linestyle='dashed')
#
# circle_BS0_change = patches.Circle(BS0_loc_draw, radius=700, fill=False, color='red', linestyle='dashed')
# circle_BS1_change = patches.Circle(BS1_loc_draw, radius=300, fill=False, color='red', linestyle='dashed')
# circle_BS2_change = patches.Circle(BS2_loc_draw, radius=200, fill=False, color='red', linestyle='dashed')
# 添加圆原始大小到坐标轴
ax.add_patch(circle_BS0)
ax.add_patch(circle_BS1)
ax.add_patch(circle_BS2)
#BS0 zooming
ax.add_patch(circle_BS0_in)
ax.add_patch(circle_BS0_out)

#BS1 zooming
ax.add_patch(circle_BS1_in)
ax.add_patch(circle_BS1_out)
#BS2 zooming
ax.add_patch(circle_BS2_in)
ax.add_patch(circle_BS2_out)

# 在点旁边标注用户和基站的名称以及距离
distance = 20  # 距离点的距离

ax.text(user1_loc_draw[0] + distance, user1_loc_draw[1] + distance, 'User1', color='black', fontsize=8, ha='right', va='bottom')
ax.text(user2_loc_draw[0] + distance, user2_loc_draw[1] + distance, 'User2', color='black', fontsize=8, ha='right', va='bottom')
ax.text(user3_loc_draw[0] + distance, user3_loc_draw[1] + distance, 'User3', color='black', fontsize=8, ha='right', va='bottom')
ax.text(user4_loc_draw[0] + distance, user4_loc_draw[1] + distance, 'User4', color='black', fontsize=8, ha='right', va='bottom')
ax.text(user5_loc_draw[0] + distance, user5_loc_draw[1] + distance, 'User5', color='black', fontsize=8, ha='right', va='bottom')

ax.text(BS0_loc_draw[0] + distance, BS0_loc_draw[1] + distance, 'BS0', color='red', fontsize=8, ha='right', va='bottom')
ax.text(BS1_loc_draw[0] + distance, BS1_loc_draw[1] + distance, 'BS1', color='red', fontsize=8, ha='right', va='bottom')
ax.text(BS2_loc_draw[0] + distance, BS2_loc_draw[1] + distance, 'BS2', color='red', fontsize=8, ha='right', va='bottom')


# 添加坐标轴标签
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

# 设置坐标轴范围
plt.xlim(0, 3300)
plt.ylim(0, 2700)

# 添加网格
# 添加网格
plt.grid()


# 显示图例
plt.legend()

# 显示图像
plt.show()

import math

result = math.sqrt(500**2 + 250**2)
print(result)