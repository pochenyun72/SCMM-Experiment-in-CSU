# 基于概率生成模型的二分类任务：确定一个人是否年收入超过5万美元。
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from numpy.linalg import inv

# 1. 数据预处理
# 1.1 读取数据
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
# 1.2 输入数据格式化处理
# 1.2.1 去除字符串数值前面的空格
str_cols = [1, 3, 5, 6, 7, 8, 9, 13, 14]
for col in str_cols:
    train_data.iloc[:, col] = train_data.iloc[:, col].map(lambda x: x.strip())
    if col != 14:
        test_data.iloc[:, col] = test_data.iloc[:, col].map(lambda x: x.strip())

# 1.2.2 将?字符串替换
# train_data['workclass'][train_data['workclass'] == "?"] = 'Private'
# test_data['workclass'][train_data['workclass'] == "?"] = 'Private'
# train_data['occupation'][train_data['occupation'] == "?"] = 'other'
# test_data['occupation'][test_data['occupation'] == "?"] = 'other'

train_data.loc[train_data['workclass'].eq("?"), 'workclass'] = 'Private'
test_data.loc[test_data['workclass'].eq("?"), 'workclass'] = 'Private'
train_data.loc[train_data['occupation'].eq("?"), 'occupation'] = 'other'
test_data.loc[test_data['occupation'].eq("?"), 'occupation'] = 'other'
# 1.2.3 对字符数据进行编码
# 训练集处理
# 放置每一列的encoder
train_label_encoder = []
train_encoded_set = np.empty(train_data.shape)
for col in range(train_data.shape[1]):
    encoder = None
    # 字符型数据
    if train_data.iloc[:, col].dtype == object:
        encoder = LabelEncoder()
        train_encoded_set[:, col] = encoder.fit_transform(train_data.iloc[:, col])
    # 数值型数据
    else:
        train_encoded_set[:, col] = train_data.iloc[:, col]
    train_label_encoder.append(encoder)

train_encoded_data = train_encoded_set

# 测试集处理
# 放置每一列的encoder
test_label_encoder = []
test_encoded_set = np.empty(test_data.shape)
for col in range(test_data.shape[1]):
    encoder = None
    # 字符型数据
    if test_data.iloc[:, col].dtype == object:
        encoder = LabelEncoder()
        test_encoded_set[:, col] = encoder.fit_transform(test_data.iloc[:, col])
    # 数值型数据
    else:
        test_encoded_set[:, col] = test_data.iloc[:, col]
    test_label_encoder.append(encoder)

test_encoded_data = test_encoded_set

# 1.3 划分训练集为训练集和验证集
X, y = train_encoded_data[:, :-1], train_encoded_data[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=50)

# 2. 加载数据以及标准化
train_mean = np.mean(X_train, axis=0).reshape(1, -1)
train_std = np.std(X_train, axis=0).reshape(1, -1)

train_data = X_train

# 3. 实现后验概率模型
class_0_id = []
class_1_id = []
for i in range(len(y_train)):
    if y_train[i] == 0:
        class_0_id.append(i)
    else:
        class_1_id.append(i)

class_0 = train_data[class_0_id]
class_1 = train_data[class_1_id]

n = class_0.shape[1]
cov_0 = np.zeros((n, n))
cov_1 = np.zeros((n, n))

# 求两类别的均值，以及共享的协方差
mean_0 = np.mean(class_0, axis=0).reshape(1, -1)
mean_1 = np.mean(class_1, axis=0).reshape(1, -1)

for i in range(class_0.shape[0]):
    cov_0 += np.dot(np.transpose(class_0[i] - mean_0), (class_0[i] - mean_0)) / class_0.shape[0]

for i in range(class_1.shape[0]):
    cov_1 += np.dot(np.transpose(class_1[i] - mean_1), (class_1[i] - mean_1)) / class_1.shape[0]

cov = (cov_0 * class_0.shape[0] + cov_1 * class_1.shape[0]) / (class_0.shape[0] + class_1.shape[0])

w = np.transpose(((mean_0 - mean_1)).dot(inv(cov)))
b = (-0.5) * (mean_0).dot(inv(cov)).dot(mean_0.T) + 0.5 * (mean_1).dot(inv(cov)).dot(mean_1.T) + np.log(
    float(class_0.shape[0]) / class_1.shape[0])

# 4. 利用模型对验证集预测
val_array = np.empty([X_val.shape[0], 1], dtype=float)
for i in range(X_val.shape[0]):
    z = X_val[i, :].dot(w) + b
    z *= (-1)
    val_array[i][0] = 1 / (1 + np.exp(z))
val_result = np.clip(val_array, 1e-8, 1 - (1e-8))
# 将预测结果转换为0、1
val_answser = np.ones([val_result.shape[0], 1], dtype=int)
for i in range(val_result.shape[0]):
    if val_result[i] > 0.5:
        val_answser[i] = 0

# 计算验证集的精确度
right_num = 0
for i in range(len(val_answser)):
    if val_answser[i] == y_val[i]:
        right_num += 1
print(right_num / len(val_answser))

# 5. 对测试集进行预测
loss = 1e-7
test_array = np.empty([test_encoded_data.shape[0], 1], dtype=float)
for i in range(test_encoded_data.shape[0]):
    z = test_encoded_data[i, :].dot(w) + b
    z *= (-1)
    test_array[i][0] = 1 / (1 + np.exp(z))
test_result = np.clip(test_array, loss, 1 - loss)
test_answer = np.ones([test_result.shape[0], 1], dtype=int)
for i in range(test_result.shape[0]):
    if test_result[i] > 0.5:
        test_answer[i] = 0

# 6. 保存预测结果到文件中
# 6.1 仅保存结果
predict_result_file = open('./predict.csv', 'w', newline='')
writer = csv.writer(predict_result_file)
writer.writerow(('id', 'label'))
for i in range(test_answer.shape[0]):
    writer.writerow([i + 1, test_answer[i][0]])
predict_result_file.close()

# 6.2 保存预测结果到原测试文件中
# 修改0、1值为<=50K、>50K
predict_answer = []
for i in range(len(test_answer)):
    if test_answer[i] == 0:
        predict_answer.append(' <=50K')
    else:
        predict_answer.append(' >50K')

# 保存预测结果到文件中
source_file = './data/test.csv'  # person.csv包括id,name,age三个列
predict_file = pd.read_csv(source_file, low_memory=False)  # 读取csv,设置low_memory=False防止内存不够时报警告
predict_file['income'] = predict_answer  # 增加新的列company

# 以下保存指定的列到新的csv文件，index=0表示不为每一行自动编号，header=1表示行首有字段名称
predict_file.to_csv('./predict_detail.csv', index=0, header=1)
