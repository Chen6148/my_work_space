from sklearn.linear_model import LinearRegression  # 线性拟合模型
import pandas as pd
import numpy as np
import json

# ------------------------------
# 初始化
#  1.读取电站样板逆变器配置信息
#  2.读取电站前60天运行日志
#  3.获取电站逆变器日发电量数据
# ------------------------------

dic = open("config/weather_info.json",'r')  # 读取样板逆变器配置信息
CONFIG_WEATHER = json.load(dic)
df_inv = pd.read_csv("datas/inverter_data_day.csv")  # 读取日发电量数据
df_log = pd.read_csv("datas/station_run_log.csv")  # 读取运行日志

# ------------------------------
# 计算逻辑
# 获取电站当天逆变器的日发电数据
# 限电损失电量自动计算逻辑（样板逆变器法，容量折算法）
# ------------------------------

def calculate_by_template(df_input):
    df = df_input
    
def calculate_by_capacity(df_input):
    df = df_input

    filter1 = now_data['device_name'].isin(config)
    df = now_data[filter1]
    if df.empty:  # 判断电站是否存在样板逆变器
        print("该时该电站没有样板逆变器，无法计算限电损失电量")
        return None
    temp_gener = df["daily_gener"].values.tolist()
    temp_capacity = df["device_capacity"].values.tolist()
    station_capacity = now_data["device_capacity"].values.tolist()
    station_power = sum(now_data["daily_gener"].values.tolist())
    theory_power = sum(temp_gener)*(sum(station_capacity)/sum(temp_capacity))
    loss_power = theory_power - station_power
    
    return loss_power
