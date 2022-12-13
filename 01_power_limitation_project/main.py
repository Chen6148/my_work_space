import json
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # 线性拟合模型

# ------------------------------
# 初始化
#  1.读取电站样板逆变器配置信息
#  2.读取电站前60天运行日志
#  3.读取电站逆变器日发电量数据
# ------------------------------

dic = open("config/weather_info.json",'r')  # 读取样板逆变器配置信息
TEM_INV = json.load(dic)
dic = open("config/weather_info.json",'r')  # 读取样板逆变器配置信息
WEATHER_INFO = json.load(dic)
df_inv = pd.read_csv("datas/inverter_data_day.csv")  # 读取日发电量数据
df_log = pd.read_csv("datas/station_run_log.csv")  # 读取运行日志

# ------------------------------
# 样板逆变器法
# 获取非限电日和限电日逆变器发电量数据
# dataA:限电日逆变器发电量数据
# dataB:非限电日逆变器发电量数据
# ------------------------------

def calculate_by_template(dataA,dataB):
    df = dataA
    df1 = dataB
    dic1 = dict(zip(df1['device_name'],df1['daily_gener']))
    df['daily_gener_n'] = df['device_name'].apply(lambda x: dic1[x])
    df['eq_hours'] = round((df['daily_gener']/df['device_capacity']),2)
    df['eq_hours_n'] = round((df['daily_gener_n']/df['device_capacity']),2)
    dic_l = dict(zip(df['device_name'],df['eq_hours']))
    dic_n = dict(zip(df['device_name'],df['eq_hours_n']))
    list_l = [dic_l[i] for i in TEM_INV[station_name]]
    list_n = [dic_n[i] for i in TEM_INV[station_name]]
    val_l = np.mean(list_l)
    val_n = np.mean(list_n) 
    df['ratio'] = (df['eq_hours']/val_l)
    df['ratio_n'] = (df['eq_hours_n']/val_n)
    df['deviate'] = df['ratio_n'] - df['ratio']
    actual_lists = [val_l*df.at[i,'ratio_n']*df.at[i,'device_capacity'] for i in range(len(df)) if df.at[i,'deviate'] > 0.0]
    theory_lists = [df.at[i,'daily_gener'] for i in range(len(df)) if df.at[i,'deviate'] > 0.0]
    loss_power = sum(theory_lists) - sum(actual_lists)
    return loss_power

# ------------------------------
# 容量折算法
# 通过限电日样板逆变器发电量折算到全站装机容量
# dataA:限电日逆变器发电量数据
# ------------------------------

def calculate_by_capacity(dataA):
    df = dataA
    dic1 = dict(zip(df['device_name'],df['daily_gener']))
    list1 = [dic1[i] for i in TEM_INV[station_name]]
    dic2 = dict(zip(df['device_name'],df['device_capacity']))
    list2 = [dic2[i] for i in TEM_INV[station_name]]
    list3 = df['device_capacity'].values.tolist()
    actual_lists = df['daily_gener'].values.tolist()
    theory_power = sum(list1) * (sum(list3)/sum(list2))
    loss_power = theory_power - sum(actual_lists)
    return loss_power

# ------------------------------
# 判断逻辑
# 寻找非限电
# ------------------------------

def calculate_by_template(df_input):
    df = df_input
    
    