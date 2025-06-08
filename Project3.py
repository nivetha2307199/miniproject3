#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from sqlalchemy.exc import OperationalError
import pandas as pd
index=0
d=[]
t=[]
gap=[]
grp=[]
v=[]
gi=[]
sm1=[]
sm2=[]
sm3=[]
with open('household_power_consumption.txt') as txt_file:
    for data in txt_file:
        items=""
        if(index!=0):
            items=data.split(";")
            d.append(items[0])
            t.append(items[1])
            gap.append(items[2])
            grp.append(items[3])
            v.append(items[4])
            gi.append(items[5])
            sm1.append(items[6])
            sm2.append(items[7])
            sm3.append(items[8])
        index+=1
password = quote_plus("$elva_30")  # encode special chars
engine = create_engine(f"mysql+pymysql://root:{password}@localhost/power_house")
try:
    dt={"Date":d,"Time":t,"Global_active_power":gap,"Global_reactive_power":grp,
        "Voltage":v,"Global_intensity":gi,"sub_metering_1":sm1,"sub_metering_2":sm2,"sub_metering_3":sm3}
    df=pd.DataFrame(dt)
    df.to_sql('dataset',engine,index=False)
    print("successfully inserted")
except OperationalError as e:
    print("OperationalError occurred:", e)


# In[ ]:




