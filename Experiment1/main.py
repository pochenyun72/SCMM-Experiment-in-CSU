# 导入相关库
import pandas as pd
import numpy as np
import csv

# 读入数据
data = pd.read_csv('./data/train.csv', encoding='utf-8')

# 数据预处理
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# 按月分割数据
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# 分割x和y
x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9]\
                .reshape(1, -1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value
# print(x)
# print(y)

# 对x标准化
mean_x = np.mean(x, axis=0)  # 18 * 9
std_x = np.std(x, axis=0)  # 18 * 9
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# 训练模型并保存权重
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x2 = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
learning_rate = 2
iter_time = 10000
adagrad = np.zeros([dim, 1])
eps = 1e-7
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x2, w) - y, 2)) / 471 / 12)  # rmse
    if t % 100 == 0:
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x2.transpose(), np.dot(x2, w) - y)  # dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / (np.sqrt(adagrad) + eps)

np.save('weight.npy', w)

# 导入测试数据test.csv
testData = pd.read_csv('./data/test.csv', header=None, encoding='utf-8')
test_data = testData.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18 * 9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)

# 对test的x进行预测，得到预测值ans_y
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
# 加一个预处理<0的都变成0
for i in range(240):
    if ans_y[i][0] < 0:
        ans_y[i][0] = 0
    else:
        ans_y[i][0] = np.round(ans_y[i][0])

# 保存为csv文件
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
