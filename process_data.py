import pandas as pd
import numpy as np


# DataFrame 读取1,2,9,10列 从第0列开始
# IO = "./data/testdata/test.xlsx"
IO = './data/pollution_data.xlsx'
sheet = pd.read_excel(io=IO, usecols=[1, 2, 9, 10])

# 有 10 个站点
sites = []
for i in range(10):
    site = sheet.loc[sheet['site'] == i]
    sites.append(site)

for i in range(10):
    sites[i] = sites[i].replace(-1, np.nan)
    # sites[i].interpolate('linear', inplace=True)  # 补全所有缺失值 会报错
    sites[i].to_excel('./data/site'+str(i)+'.xlsx', columns=['date', 'time', 'pm25'], index=False)
