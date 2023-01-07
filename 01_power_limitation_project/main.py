#!/usr/bin/env python
# coding: utf-8

# In[144]:


import datetime
import time
from dateutil.relativedelta import relativedelta
import pyodbc
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine


# In[145]:


import sys
sys.path.append('./')
import get_data


# # 1、导入数据

# In[146]:


df_station_d = pd.read_csv('./Data/station_d.csv')
df_station_d.fillna(0,inplace=True)

df_device = pd.read_csv('./Data/device.csv')
df_device=df_device[['station_code','device_full_code','template_machine','device_capacity']]

df_epower_irradiance = pd.read_csv('./Data/epower_irradiance.csv')
df_epower_irradiance.fillna(0,inplace=True)

df_inverter_today = pd.read_csv('./Data/inverter_today.csv')


# In[147]:


#获取inverter_d文件夹下的表名
file_name_list = os.listdir('./Data/inverter_d')
inverter_d=pd.DataFrame()
#拼接inverter_d文件夹内的数据
for i in file_name_list:
    aa=pd.read_csv('./Data/inverter_d/'+i)
    inverter_d=pd.concat([aa,inverter_d])
#剔除发电量为空值、负值的数据
inverter_d=inverter_d[inverter_d['inverter_actual_power']>0]
inverter_d['station_code']=inverter_d['station_code'].astype(int)
inverter_d=inverter_d.rename(columns={'device_code':'device_full_code'})


# In[149]:


#限电电站的天数
station_code=df_station_d[df_station_d['limit_power']!=0]
station_days=station_code.groupby(['station_code'])['report_date'].count().to_frame('days').sort_values(by='days').reset_index()


# In[150]:


#按照电站、日期分组聚合发电量
station_inverter_d=inverter_d.groupby(['station_code','report_date'])['inverter_actual_power'].sum().to_frame('inverter_actual_power').reset_index()

#筛选非限电的数据
df_station=df_station_d[df_station_d['limit_power']==0][['station_code','station_name','report_date','resource_value']].sort_values(by=['station_code','report_date'])
df_station=pd.merge(df_station,station_inverter_d, how='left',on=['station_code','report_date'])

#关联获取当日辐射量
df_epower_max_qx012=df_epower_irradiance[['station_code','max_qx012']]
df_epower_max_qx012['max_qx012']=df_epower_max_qx012['max_qx012']/1000
df_station=pd.merge(df_station,df_epower_max_qx012,how='left',on=['station_code'])
df_station.fillna(0,inplace=True)
#计算非限电日辐射量和当日辐射量偏差率
df_station['qx_deviation']=abs((df_station['resource_value']-df_station['max_qx012'])/df_station['max_qx012'])


# # 2、方案一、样板逆变器法

# In[151]:


#================================================2.1 电站限电天数大于30天================================================


# In[152]:


#筛选限电天数大于30天的电站
station_30=station_days[station_days['days']>30]['station_code'].values.tolist()
df_30=df_station[df_station['station_code'].isin(station_30)]
#电站去重
station=df_30.groupby(['station_code'])['station_code'].count().to_frame('inverter_actual_power').reset_index()


# In[153]:


#拟合线性模型
def Linear(station_code,data):
    df=pd.DataFrame()
    for i in station_code['station_code']:
        a=data[data['station_code']==i].reset_index(drop=True)
        X=a[['resource_value']]
        Y=a[['inverter_actual_power']]
        #训练线性模型
        model = LinearRegression()
        model.fit(X,Y)
        r_square=model.score(X,Y)#R方
        y_pred = model.predict(X)
        y_pred=pd.DataFrame(y_pred,columns=['predict_power'])
        a['r_square']=r_square
        b=pd.concat([a,y_pred],axis=1)
        df=pd.concat([b,df])
    df=df.reset_index(drop=True)
    return df
df=Linear(station,df_30)


# In[154]:


#计算预拟合模型得到的发电量和实际发电量偏差率
df['power_deviation']=abs((df['predict_power']-df['inverter_actual_power'])/df['inverter_actual_power'])


# In[155]:


#筛选出R方＞0.9、发电量偏差≤5%、辐射量偏差≤5%的数据,并选取与当日辐照度偏差最小的非限电日
a=df[(df['r_square']>0.9)&(df['power_deviation']<=0.05)&(df['qx_deviation']<=0.05)]
#a=df[(df['r_square']>0.9)&(df['power_deviation']<=0.05)]
b=a.sort_values(by='power_deviation').reset_index(drop=True)
c=b.groupby(['station_code']).head(1).reset_index(drop=True)
unlimit_date_30=c[['station_code','report_date']]
unlimit_date_30['type']=30


# In[156]:


#================================================2.2 电站限电天数大于7天，小于30天================================================


# In[157]:


#筛选限电天数为大于7天，小于30天的电站
station_7=station_days[(station_days['days']<=30)&(station_days['days']>7)]['station_code'].values.tolist()
#2.1中剩余的电站
station_other_30=[]
for i in station_30:
    if i not in unlimit_date_30['station_code'].tolist():
        station_other_30.append(i)
        
station_7.extend(station_other_30)


# In[158]:


##筛选限电天数为大于7天，小于30天的数据,并选取与当日辐照度偏差最小的7个非限电日
df_7=df_station[df_station['station_code'].isin(station_7)]
b=df_7.sort_values(by=['qx_deviation','station_code','report_date'],ascending=[True,True,False]).reset_index(drop=True)
c=b.groupby(['station_code']).head(7).reset_index(drop=True)
unlimit_date_7=c[['station_code','report_date']].sort_values(by=['station_code','report_date']).reset_index(drop=True)
unlimit_date_7['type']=7


# In[160]:


#================================================2.3 电站限电天数小于7天================================================


# In[161]:


#筛选限电天数为小于7天的电站
station_0=station_days[station_days['days']<=7]['station_code'].values.tolist()
df_0=df_station[df_station['station_code'].isin(station_0)]


# In[162]:


##选取与当日辐照度偏差最小的非限电日，并且辐照度偏差≤5%的电站
a=df_0.sort_values(by=['qx_deviation','station_code']).reset_index(drop=True)
b=a.groupby(['station_code']).head(1).reset_index(drop=True)
unlimit_date_1=b[b['qx_deviation']<=0.05][['station_code','report_date']].sort_values(by=['station_code','report_date']).reset_index(drop=True)
unlimit_station_1=unlimit_date_1['station_code'].values.tolist()


# In[163]:


##选取相邻的非限电日
a=df_0[~df_0['station_code'].isin(unlimit_station_1)]
b=a.sort_values(by=['station_code','report_date'],ascending=False).reset_index(drop=True)
c=b.groupby(['station_code']).head(1).reset_index(drop=True)
unlimit_date_2=c[['station_code','report_date']].sort_values(by=['station_code','report_date']).reset_index(drop=True)
unlimit_date_0=pd.concat([unlimit_date_1,unlimit_date_2]).reset_index(drop=True)
unlimit_date_0['type']=0


# In[165]:


#================================================2.4 计算电站限电损失================================================


# In[166]:



#计算非限电日样板逆变器等效小时数
unlimit_date=pd.concat([unlimit_date_0,unlimit_date_7,unlimit_date_30]).reset_index(drop=True)
inverter_his=pd.merge(inverter_d,unlimit_date,how='inner',on=['station_code','report_date'])
inverter_his=inverter_his[['station_code','device_full_code','report_date','device_capacity','inverter_actual_power']]
df_device1=df_device[['station_code','device_full_code','template_machine']]
inverter_his=pd.merge(inverter_his,df_device1,how='left',on=['station_code','device_full_code'])
inverter_his_1=inverter_his[inverter_his['template_machine']==1]#筛选样板逆变器
inverter_his_1=inverter_his_1.groupby(['station_code','report_date'])['device_capacity','inverter_actual_power'].sum().reset_index(drop=False)
inverter_his_1['inverter_h_his1']=inverter_his_1['inverter_actual_power']/inverter_his_1['device_capacity']
inverter_his_1=inverter_his_1[['station_code','report_date','inverter_h_his1']]
#计算非限电日的非样板逆变器等效小时数
inverter_his_0=inverter_his[inverter_his['template_machine']==0]#筛选非样板逆变器
inverter_his_0['inverter_h_his0']=inverter_his_0['inverter_actual_power']/inverter_his_0['device_capacity']
#计算当日样板逆变器等效小时
unlimit_date_x=unlimit_date[['station_code','type']]
inverter_today_h=pd.merge(df_inverter_today,df_device1,how='left',on=['station_code','device_full_code'])
#inverter_today_h_1=pd.merge(inverter_today_h,unlimit_date_x,how='inner',on=['station_code'])
inverter_today_h_1=inverter_today_h[inverter_today_h['template_machine']==1]#筛选样板逆变器
inverter_today_h_1=inverter_today_h_1.groupby(['station_code'])['device_capacity','daily_gen'].sum().reset_index(drop=False)
inverter_today_h_1['inverter_h_today1']=inverter_today_h_1['daily_gen']/inverter_today_h_1['device_capacity']
inverter_today_h_1=inverter_today_h_1[['station_code','inverter_h_today1']]


# In[167]:


#计算当日非样板逆变器等效小时
inverter_today_h_0=pd.merge(inverter_his_0,inverter_his_1,how='left',on=['station_code','report_date'])
inverter_today_h_0=inverter_today_h_0.groupby(['station_code','device_full_code','template_machine'])['inverter_h_his0','inverter_h_his1'].mean().reset_index(drop=False)
inverter_today_h_0=pd.merge(inverter_today_h_0,inverter_today_h_1,how='left',on=['station_code'])
inverter_today_h_0['inverter_today_h0']=inverter_today_h_0['inverter_h_his0']/inverter_today_h_0['inverter_h_his1']*inverter_today_h_0['inverter_h_today1']
#计算当日非样板逆变器的应发电量
a=df_device[['station_code','device_full_code','device_capacity']]
inverter_today_h_0=pd.merge(inverter_today_h_0,a,how='left',on=['station_code','device_full_code'])
inverter_today_h_0['inverter_predict_power']=inverter_today_h_0['device_capacity']*inverter_today_h_0['inverter_today_h0']
inverter_today_h_0=inverter_today_h_0[['station_code','device_full_code','inverter_predict_power']]

#计算全站限电损失电量
result1=pd.merge(inverter_today_h,inverter_today_h_0,how='left',on=['station_code','device_full_code'])
result1=result1[result1['template_machine']==0]
result1.fillna(0,inplace=True)
result1=result1.groupby(['station_code','report_date'])['daily_gen','inverter_predict_power'].sum().reset_index(drop=False)
result1['limit_power']=result1['inverter_predict_power']-result1['daily_gen']


# # 3、方案二、容量折算法

# In[175]:


#筛选限电天数等于60天的电站
station_60=station_days[station_days['days']==60]['station_code'].values.tolist()


# In[169]:


df_60=inverter_today_h[inverter_today_h['station_code'].isin(station_60)]
device_capacity_all=df_60.groupby(['station_code'])['device_capacity'].sum().to_frame('device_capacity_all').reset_index(drop=False)
device_capacity_all['template_machine']=1
df_60=pd.merge(df_60,device_capacity_all,how='left',on=['station_code','template_machine'])
df_60.fillna(0,inplace=True)
df_60['inverter_predict_power']=df_60['device_capacity_all']/df_60['device_capacity']*df_60['daily_gen']
result2=df_60.groupby(['station_code','report_date'])['daily_gen','inverter_predict_power'].sum().reset_index(drop=False)
result2['limit_power']=result2['inverter_predict_power']-result2['daily_gen']


# In[173]:


#输出结果
result=pd.concat([result1,result2]).reset_index(drop=True)
result=result[['station_code','report_date','limit_power']]
result=result[result['limit_power']>0]
now_time = datetime.datetime.now()
result['updatetime']=now_time
today = datetime.date.today()
sql='''delete from py_limit_power  where report_date=\'%s\''''%today
engine = create_engine('mssql+pymssql://SDZRPT:84@6d752AE@10.121.1.137/epower_report')
engine.execute(sql)
result.to_sql('py_limit_power', engine, if_exists='append', index=False)

