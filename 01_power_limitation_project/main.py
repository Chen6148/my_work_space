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
# 计算逻辑
# 获取电站当天逆变器的日发电数据
# 限电损失电量自动计算逻辑（样板逆变器法，容量折算法）
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

def calculate_by_capacity(inputData):
    df = inputData
    dic1 = dict(zip(df['device_name'],df['daily_gener']))
    list1 = [dic1[i] for i in TEM_INV[station_name]]
    dic2 = dict(zip(df['device_name'],df['device_capacity']))
    list2 = [dic2[i] for i in TEM_INV[station_name]]
    list3 = df['device_capacity'].values.tolist()
    actual_lists = df['daily_gener'].values.tolist()
    theory_power = sum(list1) * (sum(list3)/sum(list2))
    loss_power = theory_power - sum(actual_lists)
    return loss_power
    
def to_determine(station_name,now_date):
    station_name = '中卫正泰'
    now_date = '2022-07-09'

    df = df_log[(df_log['station_name'] == station_name)]
    now_dates = datetime.datetime.strptime(now_date, '%Y-%m-%d')
    last_date =now_dates - datetime.timedelta(days=60)
    last_date = datetime.datetime.strftime(last_date, '%Y-%m-%d')
    print(last_date)
    df = df[(df['date'] >= last_date) & (df['date'] <= now_date)]
    df = df[df['limit_loss_power'] == 0]
    dates = df['date'].values.tolist()
    dataX = []  # 存储辐照量
    dataY = []  # 存储发电量
    for i in dates:
        df = df_inv[(df_inv['station_name'] == station_name) & (df_inv['date'] == i)]
        X = WEATHER_INFO[station_name][i]
        Y = df['daily_gener'].sum()
        dataX.append(X)
        dataY.append(Y)
    X_train = np.array(dataX).reshape((len(dataX), 1))
    Y_train = np.array(dataY).reshape((len(dataY), 1))
    lineModel = LinearRegression()
    lineModel.fit(X_train, Y_train)
    goal = lineModel.score(X_train, Y_train)
    dateA = []
    if goal > 0.8:
        Y_predict = lineModel.predict(X_train)
        arr = abs(Y_predict-Y_train)/Y_train
        for i in range(len(arr)):
            if arr[i][0] < 0.05:dateA.append(dates[i])
    print(dateA)
    irr_1 = [WEATHER_INFO[station_name][i] for i in dateA]
    irr_2 = np.array(irr_1) - WEATHER_INFO[station_name][now_date]
    print(irr_2)
    dateB = {}
    for i in dateA:
        irr_1 = WEATHER_INFO[station_name][i]
        irr_2 =abs(WEATHER_INFO[station_name][i] - WEATHER_INFO[station_name][now_date])
        if irr_2 < 0.05:dateB.update({irr_2:i})
    print(dateB.keys(),dateB.values())
    dat = min(dateB.keys())
    print(dat,dateB[dat])