import os
import pandas as pd
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import tensorflow as tf

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 列
    f.write('NA,Pave,127500\n') # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,131400\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data,'\n')

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
print(inputs,'\n')

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs,'\n')

# x = tf.constant(inputs.to_numpy(dtype=float))
# y = tf.constant(outputs.to_numpy(dtype=float))
# print(x,'\n',y)


data1 = data.drop(data.isna().sum().idxmax(),axis=1) # 删除nan最多的列
print(data1)
