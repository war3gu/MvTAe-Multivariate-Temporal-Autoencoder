import os
import struct
import datetime
import math

def get_date_str(h1, h2) -> str:  # H1->0,1字节; H2->2,3字节;
    year = math.floor(h1 / 2048) + 2004  # 解析出年
    month = math.floor(h1 % 2048 / 100)  # 月
    day = h1 % 2048 % 100  # 日
    hour = math.floor(h2 / 60)  # 小时
    minute = h2 % 60  # 分钟
    if hour < 10:  # 如果小时小于两位, 补0
        hour = "0" + str(hour)
    if minute < 10:  # 如果分钟小于两位, 补0
        minute = "0" + str(minute)
    return str(year) + "-" + str(month) + "-" + str(day) + " " + str(hour) + ":" + str(minute)

'''
def stock_csv(fullpath, path, name):
    data = []
    with open(fullpath, 'rb') as f:
        file_object_path = os.path.join(path,name) + '.csv'
        file_object = open(file_object_path, 'w+')
        while True:
            h1 = f.read(2)
            h2 = f.read(2)

            stock_open = f.read(4)
            stock_high = f.read(4)
            stock_low = f.read(4)
            stock_close = f.read(4)
            stock_amount = f.read(4)
            stock_vol = f.read(4)
            stock_reservation = f.read(4)  # date,open,high,low,close,amount,vol,reservation
            if not h1:
                break
            h1 = struct.unpack("H", h1)
            h2 = struct.unpack("H", h2)

            stock_date = get_date_str(h1[0], h2[0])

            # 开盘价*100
            stock_open = struct.unpack("l", stock_open)
            # 最高价*100
            stock_high = struct.unpack("l", stock_high)
            # 最低价*100
            stock_low = struct.unpack("l", stock_low)
            # 收盘价*100
            stock_close = struct.unpack("l", stock_close)
            # 成交额
            stock_amount = struct.unpack("f", stock_amount)
            # 成交量
            stock_vol = struct.unpack("l", stock_vol)
            # 保留值
            stock_reservation = struct.unpack("l", stock_reservation)
            # 格式化日期
            date_format = datetime.datetime.strptime(str(stock_date[0]), '%Y%M%d')
            list = date_format.strftime('%Y-%M-%d') + "," + str(stock_open[0] / 100) + ","\
                   +str(stock_high[0] / 100.0) + "," + str(stock_low[0] / 100.0) + ","\
                   + str(stock_close[0] / 100.0) + "," + str(stock_vol[0]) + "\r\n"
            file_object.writelines(list)
        file_object.close()
'''
def stock_csv(fullpath, path, name):
    data = []
    with open(fullpath, 'rb') as f:
        file_object_path = os.path.join(path,name) + '.csv'
        file_object = open(file_object_path, 'w+')
        while True:
            li2 = f.read(32)  # 读取一个5分钟数据
            if not li2:  # 如果没有数据了，就退出
                break
            data2 = struct.unpack('HHffffllf', li2)  # 解析数据
            date_str = get_date_str(data2[0], data2[1])  # 解析日期和分时

            data2_list = list(data2)  # 将数据转成list
            data2_list[1] = date_str  # 将list二个元素更改为日期 时:分
            del (data2_list[0])  # 删除list第一个元素
            for dl in range(len(data2_list)):  # 将list中的内容都转成字符串str
                data2_list[dl] = str(data2_list[dl])
            data2_str = ",".join(data2_list)  # 将list转换成字符串str
            data2_str += "\n"  # 添加换行
            file_object.writelines(data2_str)  # 写入一行数据
        file_object.close()  # 完成数据写入




path = './line/'
listfile = os.listdir(path)
for dir in listfile:
    dd = os.path.join(path, dir)
    listfile1 = os.listdir(dd)
    for dir1 in listfile1:
        dd1 = os.path.join(dd, dir1)
        listfile2 = os.listdir(dd1)
        for dir2 in listfile2:
            if dir2.endswith("lc1") or dir2.endswith("lc5") :
                dd2 = os.path.join(dd1, dir2)
                stock_csv(dd2, dd1, dir2[:-4])


#lc1,lc5的数据量太大了，选30只从15年开始至今的数据分析一下就可以了