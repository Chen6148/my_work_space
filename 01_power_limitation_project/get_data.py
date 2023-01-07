#!/usr/bin/env python
# coding: utf-8

# In[148]:


import datetime
from dateutil.relativedelta import relativedelta
import pyodbc
import pandas as pd 
import numpy as np
import os
import clickhouse_connect
from influxdb import InfluxDBClient


# In[149]:


#创建文件夹
file = './Data'
if os.path.exists( file ):
    pass
else:
    os.mkdir( file )

file = './Data/inverter_d'
if os.path.exists( file ):
    pass
else:
    os.mkdir( file )


# In[150]:


#日期
today = datetime.date.today()
yestoday = today - relativedelta(days=1)
today_60 = today - relativedelta(days=60)
print(str(today_60)+'~'+str(today)+'时间段内')


# In[122]:


#数据库连接信息
server = '10.121.1.137' 
database = 'epower_report' 
username = 'SDZRPT' 
password = '84@6d752AE' 


# In[123]:


def query(sql):
    #连接数据库
    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = cnxn.cursor()
    cursor.execute(sql) 
    data=cursor.fetchall()
    dataname=cursor.description
    cursor.close()
    cnxn.close()
    #list转dataframe
    dataname_1=[list(i)  for i in dataname]
    dataname_f=pd.DataFrame(dataname_1)
    dataname_f=dataname_f[0]
    #print(dataname_f)
    data_1=[ list(i)  for  i in data]
    df=pd.DataFrame(data_1,columns=dataname_f)
    return df


# In[124]:


#获取近60天存在限电的电站
sql_station_code='''select  distinct station_code from   finereport_station_d  where report_date>=\'%s\' and limit_power!=0'''% today_60
station_code=query(sql_station_code)
station_count=station_code.iloc[:,0].size
station_code=station_code['station_code'].values.tolist()
print('存在限电的电站个数：'+str(station_count))


# # 1、获取限电电站运行日志、设备信息

# In[125]:


#sql
sql_station_d='''select  * from finereport_station_d where  check_status=1 and  report_date>='{}' 
and station_code in ({})'''.format(today_60, ','.join(["'%s'" % station_code for station_code in station_code]))


sql_device='''select  * from epower_ori_tbl_device where device_type_code in (201,206) 
and station_code in ({})'''.format( ','.join(["'%s'" % station_code for station_code in station_code]))

#sql_epower_irradiance='''select   * from epower_203day where date='{}'
#and station_code in ({})'''.format(today, ','.join(["'%s'" % station_code for station_code in station_code]))

#sql_solargis='''select   * from inspect_solargis_data where collect_date>='{}'
#and station_code in ({})'''.format(today_60, ','.join(["'%s'" % station_code for station_code in station_code]))


# In[126]:


station_d=query(sql_station_d)
device=query(sql_device)
#epower_irradiance=query(sql_epower_irradiance)
#solargis=query(sql_solargis)
#导出csv
station_d.to_csv('./Data/station_d.csv', sep=',', header=True, index=True)
device.to_csv('./Data/device.csv', sep=',', header=True, index=True)
#epower_irradiance.to_csv('./Data/epower_irradiance.csv', sep=',', header=True, index=True)
#solargis.to_csv('./Data/solargis.csv', sep=',', header=True, index=True)


# # 2、获取历史逆变器发电量数据（截止昨日）

# In[127]:


#日期范围
date_range=pd.date_range(today_60,yestoday)
date_range=pd.DataFrame(date_range,columns=['date'])
date_range['date']=date_range['date'].astype( str )


# In[128]:


#获取inverter_d文件夹下的表名
file_name_list = os.listdir('./Data/inverter_d')
#格式处理
file_name = [i.replace('.csv','') for i in file_name_list]
file_name = [i.replace('inverter_d_','') for i in file_name]
file_name=pd.DataFrame(file_name,columns=['date'])
file_name['date']=pd.to_datetime(file_name['date'])
file_name['date']=file_name['date'].astype( str )
file_name_list=file_name['date'].values.tolist()


# In[129]:


#删除小于today_60的逆变器日数据
for i in file_name_list:
     if i < str(today_60):
          i_date=datetime.date(*map(int, i.split('-'))) #转date格式  
          os.remove('./Data/inverter_d/inverter_d_'+i_date.strftime("%Y%m%d")+'.csv')


# In[130]:


def  inverter_query(start_date,end_date,file_name_list):
    #日期范围
    date_range=pd.date_range(start_date,end_date)
    date_range=pd.DataFrame(date_range,columns=['date'])
    date_range['date']=date_range['date'].astype( str )
    #判断inveter_d数据缺失日期
    lost_date=date_range[~date_range['date'].isin(file_name_list)]  
    #lost_date['date']=lost_date['date'].astype( str )
    lost_date=lost_date['date'].values.tolist() 
    #循环遍历
    for i in lost_date:
        i_date=datetime.date(*map(int, i.split('-'))) #转date格式
        sql_inverter_d='''select  * from epower_inverter_data_day  where report_date=\'%s\''''%i
        df=query(sql_inverter_d)
        df.to_csv('./Data/inverter_d/inverter_d_'+i_date.strftime("%Y%m%d")+'.csv', sep=',', header=True, index=True)


# In[131]:


import datetime
inverter_query(today_60,yestoday,file_name_list)


# # 3、获取当日逆变器发电量数据

# In[155]:


#连接clickhouse
client = clickhouse_connect.get_client(host='10.121.1.219', port='8123')
sql = 'select  data_date,station_code,device_full_code,station_name,device_name,device_capacity,daily_gen from ads_astronergy_dz.ads_dz_inverter_data_d where data_date=\'%s\''%today
df = client.query(sql).result_set


# In[157]:


col_name=['report_date','station_code','device_full_code','station_name','device_name','device_capacity','daily_gen']
inverter_today=pd.DataFrame(df,columns=col_name)
inverter_today.to_csv('./Data/inverter_today.csv', sep=',', header=True, index=True)


# # 4、获取当日辐照度数据

# In[134]:


def conn_db(a):  
    #获取数据库列表
    conn_db = InfluxDBClient(host='10.121.2.6', port=18086, username='admin', password='admin',database=a)
    database=conn_db.get_list_database() 
    database=pd.DataFrame(database)
    database=database[~database['name'].isin(['_internal'])] 
    database_list=database['name'].tolist() 
    return database_list


# In[135]:


from datetime import datetime, timedelta
#获取influxdb气象数据
def epower_irradiance(database_list):  
     #获取气象站表名   
    table=pd.DataFrame()
    for i in database_list:
        conn_db = InfluxDBClient(host='10.121.2.6', port=18086, username='admin', password='admin',database=i)
        result = conn_db.query('show measurements')  
        table_name = list(result.get_points(measurement='measurements'))
        table_name=pd.DataFrame(table_name)
        table_name=table_name[table_name['name'].str.contains('203PT10M$')]
        table_name_list=table_name['name'].tolist()
        #获取气象数据
        a=pd.DataFrame()
        for j in table_name_list:
            sql_ir='''SELECT * from minute.{}   WHERE time > now() - 1d'''.format(j)
            result = conn_db.query(sql_ir)
            result = list(result.get_points(measurement=j))
            result=pd.DataFrame(result)
            if result.empty or 'QX012_Max' not in result:
                pass
            else:
                #时间变换
                result['time']=result['time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%dT%H:%M:%SZ'))
                result['time'] = result['time'] + timedelta(hours=8)
                result['date'] = result['time'].map(lambda x:x.strftime('%Y-%m-%d'))
                result['station_code']=j.replace('S','').replace('M203PT10M','')
                result=result[result['time']>=str(today)]
                result=result.groupby(['station_code','deviceCode','date'])['QX012_Max'].max().to_frame('max_qx012').reset_index()
                a=pd.concat([result,a],ignore_index = True)
        table=pd.concat([a,table],ignore_index = True)
    return table


# In[136]:


database_list=conn_db('s0')
epower_ir=epower_irradiance(database_list)
epower_ir.to_csv('./Data/epower_irradiance.csv', sep=',', header=True, index=True)


# In[ ]:




