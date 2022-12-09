from sklearn.linear_model import LinearRegression  # 线性拟合模型
import pandas as pd
import numpy as np
import json

# ------------------------------
# 初始化
#  1.读取电站样板逆变器配置信息
#  2.读取电站前60天运行日志
#  3.读取电站逆变器日发电量数据
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
    
    