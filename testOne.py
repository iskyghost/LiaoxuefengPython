def my_abs(x):
    if x >= 0:
        return x
    else:
        return -x

def nop():
    pass

def my_abs(x):
    if not isinstance(x, (int, float)):
        raise TypeError('bad operand type')
    if x >= 0:
        return x
    else:
        return -x

import math
def move(x, y, step, angle = 0):
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)
    return nx, ny
x, y = move(100, 100, 60, math.pi / 6)
print(x, y)

def power(x):
    return x * x

def power(x, n):
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s

def enroll(name, gender, age = 6, city = 'beijing'):
    print('name:', name)
    print('gender:', gender)
    print('age:', age)
    print('city:', city)

def add_end(L = []):
    L.append('END')
    return L

def add_end(L = None):
    if L is None:
        L = []
    L.append('END')
    return L

def calc(numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
# for example: calc([1, 2, 3])
# use variable param
calc(1, 2, 3)

def calc(*numbers):
    sum = 0
    for n in nubers:
        sum = sum + n * n
    return sum
# calc(1,2)

nums = [1, 2, 3]
calc(nums[0], nums[1], nusm[2])
# that's too tedious

nums = [1, 2, 3]
calc(*nums)

def person(name, age, **kw):
    print('name:', name, 'age:', age, 'other:', kw)
# person('bob', 35, city = 'beijing')
# name : bob age: 35 other: {'city': 'beijing'}
def person(name, age, **kw):
    if 'city' in kw:
        pass
    if 'job' in kw:
    print('name:', name, 'age:', age, 'other:', kw)

def fact(n):
    if n == 1:
        return 1
    return n * fact(n-1)

def fact(n):
    return fact_iter(n, 1)
def fact_iter(num, product):
    if num == 1:
        return product
    return fact_iter(num - 1, num * product)

r = []
n= 3
for i in range(n):
    r.append(L[i])

L = list(range(100))

d = {'a': 1, 'b' : 2, 'c': 3}
for key in d:
    print(key)

for ch in 'ABC':
    print(ch)

for value in d.values():
    print(value)

for k, v in d.items():
    print('key:', k, 'value:', v)

# is iterable？
from collections import Iterable
isinstance('abc', Iterable) #is str iterable?


for i, value in enumerate(['a', 'b', 'c']):
    print(i, value)

for x, y in [(1,1), (2, 2), (3, 3)]:
    print(x, y)

L = []
for x in range(1, 11):
    L.append(x * x)

[x * x for x in range(1, 11)]
[x * x for x in range(1, 11) if x % 2 == 0]
[d for d in os.listdir(',')]

def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        a, b = b, a + b
        n = n + 1
        return 'done'

from collections import Iterator
isinstance((x for x in range(10)), Iterator)

isinstance(iter('abc'), Iterator)

it = iter([1,2,3,4,5])
while True:
    try:
        x = next(it)
    except StopIteration:
        break

def f(x):
    return x * x
L = [1,2,3,4,5]
r = map(f, L)
list(r)

list(map(str, [1,2,3,4,5,6,7,8,9]))#change number to char

from functools import reduce
def fn(x, y):
    return x * 10 + y

def not_empty(s):
    return s and s.strip()

def _odd_iter():
    n = 1
    while True:
        n = n + 2
        yield n

def primes():
    yield 2
    it = _odd_iter()
    while True:
        n = next(it)
        yield n
        it = filter(_not_divisible(n), it)

for n in primes():
    if n < 1000:
        print(n)
    else:
        break

sorted([1,2,-4,34], key = abs)

def calc_sum(*args):
    ax = 0
    for n in args:
        ax = ax + n

        return ax
def lazy_sum(*arsg):
    def sum():

list(map(lambda x : x * x, [1,2,3,4,5,6,7]))
list(lambda x : x % 2 == 1, range(1, 20))

# def fun():
#     print('hello')
#
# f = fun
# f()
def log(func):
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper

a = float(input("输入A:"))
b = float(input("输入B:"))
c = float(input("输入C:"))
delta = B**2 - 4 *A * C
if delta < 0:
    print("无解")
elif delta == 0:
    x = B / (-2 * A)
    print("x1=x2=", x)
else:
    x1 = (B + delta**0.5) / (-2 * A)
    x2 = (B - delta**0.5) / (-2 * A)
print('x1= ', x1)
print('x2= ', x2)

n = int(input("Please input a number: "))
i = 1
sum = 0
for i in range(n + 1):
    sum += i
print("The sum is : %d " % sum )

def args_input():
    try:
        a = float(input("输入A:"))
        b = float(input("输入B:"))
        c = float(input("输入C:"))
        return a, b, c
    except:
        print("Please input correct type:")
        return args_input()
def get_delta(a, b, c):
    return B**2 - 4*A*C
def solve():
    a, b, c 

from pandas import Series
x = Series(['a', True, 1], index = ['first', 'second', 'third'])
x[1] #access by index num
x['second'] #access by index name

x[3] #it's wrong because overstep the boundary
x.append(['2']) # can'n append single element

n = Series(['2'])
x = append(n) #what we can do is just append a series in the end

x = x.append(n) #need a new variable to carry the change Series

2 in x.values #detect a number whether exits

x[1:3] #slice up
x.drop(0) #delete
x.index[2] #find index name relate of index number
etc...

DataFram数据框用于存储多行和多列的数据集合，是Series的容器，类似于excel表格，操作（增删改查）
from pandas import Series
from pandas import DataFrame
df = DataFrame({'age':Series([26,29,24]),'name':Series(['Ken', 'Jerry', 'Ben'])})
print(df)

df['age'] #access column
def[1:3] #access one to two row dataset

数据的存在形式:文件（csv,Excel,txt),数据库（Mysql，Access，SqlServer）
pandas中，常用载入函数read_csv,read_excel, read_table),若是服务器部署，还需read_sql函数，直接访问数据库
但必须配合Mysql相关的包
1、导入txt文件
read_table(file, names=[col_1, col_2,...], sep="",...)#file path, sep is 分割符，默认为空
eg:
from pandas import read_table
df = read_table(r'C:User\file.txt',sep = " ")
df.head()#默认前五项
txt要保存为 utf-8才不会报错
查看数据框df前n项使用df.head(n); 后m项数据用df.tail(m),默认5项
2、导入csv文件（逗号分隔值）
read_csv(file, names=[col_1, col_2, ...], sep="", ...)
3、导入excel文件
read_excel(file, sheetname, header=0)
eg:
from pandas import read_excel
df = read_excel(r'C:User\test.xls', sheetname = "Sheet3")
df
header取0表示以文件第一行作为表头显示，取1则把第一行丢弃，不作表头显示
4、导入MYSQL库
在anaconda3环境中加载模块：pip install pymysql
在编译器中导入：import pymysql
read_sql(sql, conn)
import pandas as pd
import pymysql
dbconn = pymysql.connect(host= "...........",
                         database= "kimbo",
                         user = "kimbo_test",
                         password= "........",
                         port=3306,
                         charset= 'utf8')
sqlcmd = "select * from table_name"
a = pd.read_sql(sqlcmd, dbconn)
dbconn.close()
b = a.head()
print(b)
#其他方法
import pymysql.cursors
import pymysql
import pandas as pd
config = {'host': '127.0.0.1',  #connect config information
          'port': 3306,
          'user': 'root',
          'password': 'root',
          'db': 'dbtest',
          'charset': 'utf-8',
          'cursorclass':pymysql.cursors.DictCursor}
conn = pymysql.connect(**config)#create connection
try:    #execute SQL statement
    with conn.cursor() as cursor:
        sql = "select * from table_name "
        cursor.execute(sql)
        result = cursor.fetchall()
finally:
    conn.close();
df = pd.DataFrame(result)
print(df.head())


1、导出csv文件
to_csv(file_path, sep=", ", index=TRUE, header= TRUE)#index表示是否导出行序号，默认TRUE，header是否导出列名，默认TRUE
from pandas import DataFrame
from pandas import Series
df = DataFrame({'age':Series([1,2,3]), 'name': Series(["Lily", "John", "Michel"])})
df.to_csv('G:\\01.csv')
df.to_csv('G:\\02.csv', index = False)

2、导出excel文件
to_excel(file_path, index = TRUE, header = TRUE)

3、导出到MySQL
to_sql(tableName, con = 数据库连接)
from pandas import DataFrame #python3.6
from pandas import Series
from sqlalchemy import create_engine
engine = create_engine("mysql + pymysql: //user: password@host: port/ databasename? charset = utf-8")#前者是账户密码，后者访问地址和端口
df = DataFrame({'age': Series([1,2,3]), 'name': Series(["Lily", "Jerry", "Michel"])})
df.to_sql(name = 'table_name',
          con = engine,
          if_exists = 'append',
          index = False,
          index_lable = False)

海量数据中存在大量不完整、不一致、有异常的数据
1、重复值的处理
duplicated(self, subset = None, keep = 'first') #return a bool of Series, if duplicate, display TRUE from it's second line
subset识别重复的列标签
keep = 'first' #except first , else label as duplicate
keep = 'last' #except last, else label as duplicate
keep = 'False' #all same data label as duplicate
#no assign specify param, so it's default
# if you want to specify attribute name, do like this:
frame.drop_duplictes(['state'])

2、缺失值的处理
#Pandas use NaN represent float and None float array's lacking/deficiency data
# use '.isnull' and '.notnull' to verdict/judge
df = read_excel(r'G:test.xlsx', sheetname = 'Sheet_Two')
df.isnull()
df.notnull()
缺失数据的处理：数据对齐、删除对应行、不处理
dropna() #delete the row of value equals NULL
df.fillna() #use else value instead NaN
df.fiilna(method = 'pad') #use prior value instead lacking
df.fiilna(methed = 'bfill') #use rear value instead
df.fillna(df.mean()) #use mean to instead
df.fiilna(df.mean()['填补列名': '计算均值的列名'])
df.fillna({'colName_one': value1, 'colName_two': value2}) #transfer a dictionary, fill different value for differ col
strip() #delete specify chars on the left and right
df['name'].str.rstrip() #delete the chars on the right
df['name'].str.lstrip('n') #delete the char 'n' on the left eg:nine-->ine

1、字段抽取：抽出某列上指定位置的数据做成新的列
slice(start, stop)
df = read_excel()
df.head() #display the first five lines
df['电话'] = df['电话'].astype(str) #type cast/conversion #电话11位，前三位是品牌，中间四位是地区，后四位是手机号
bands = df.['电话'].str.slice(0, 3) #157
areas = df.['电话'].str.slice(3, 7) #1518
tell = df.['电话'].str.slice(7, 11) #8593

2、字段拆分
split(sep, n, expand = False) #按指定字符拆分已有的字符串， sep分隔字符串的分隔符， n分割后新增的列数. expand是否展开为数据框，TRUE展开为DataFrame， False展开为Series
df['IP'].str.strip() # transfer string, delete space
newDF = df['IP'].str.split('.', 1, True) #result: 221   205.98.55
newDF.columns = ['IP1', 'IP2-4'] #为第一二列增加列名

3、重置索引：指定某列为索引
df.set_index('列名')
df = DataFrame({'age': Series([11, 13, 15]), 'name' : Series(['John', 'Lily', 'Michel'])})
df1 = df.set_index('name')
df1.ix['John'] #使用ix函数对john用户信息进行提取

4、记录抽取：根据一定条件对数据进行抽取
df[condition]
常用condition类型：比较、空置、字符匹配、逻辑运算
df[df.电话 == 13322252452]
df[df.电话 > 13500000000]
df[df.电话.between(13400000000, 13999999999)]
df[df.IP.isnull()]
df[df.IP.str.contains('222.', na = False)]

5、随机抽样：随机从数据中按照一定的行数或比例抽取数据
numpy.random.randint(start, end, num) # num is numbers of sample
df[df.电话 >= 18822256753] #single condition
df[(df.电话 >= 13422259938) & (df.电话 < 13822254373)] #改方式获取的数据切片都是DataFrame
r = numpy.random.randint(0, 10, 3) #0行到10行， 抽取3行
out>>>array([3, 4, 9])
df.loc[r, :] #抽取r行数据

6、通过索引抽取数据
使用索引名选取数据：df.loc[行标签，列标签]
df = df.set_index('name')
df.head()
df.loc['a', 'b'] #选取a到b行的数据
df.loc[:, '电话'].head() #选取电话列的数据
df.loc[[1, 2]] #抽取1、2两行

使用索引号选取数据：df.iloc[行索引号，列索引号]
df.iloc[1, 0] #取第二行第一列的值
df.iloc[[0, 2], :] #取第一行和第三行的数据
df.iloc[0: 2, :] #取第一行到第三行的数据
df.iloc[: , 1] #所有记录第二列的值，返回Series
df.iloc[1, :] #取第二行数据

7、字典数据抽取：将字典数据抽取为Dataframe
key和value各作为一列
d1 = {'a': '[1,2,3]', 'b': '[0,1,2]'}
a1 = pandas.DataFrame.from_dict(d1, orient = 'index') #tansfer dict to Dataframe, key as the index
a1.index.name = 'key' #make index rename as 'key'
b1 = a1.reset_index()
b1.columns = ['key', 'value']
>>>key  value
0   b   [0,1,2]
1   a   [1,2,3]

字典的每一个元素作为一列（同长）
d2 = {'a': [1,2,3], 'b': [4,5,6]}
a2 = DataFrame(d2)
>>>a    b
0   1   4
1   2   5
2   3   6

字典的每个元素作为一列（不同长）
d = {'one': pandas.Series([1,2,3]), 'two': pandas.Series([1,2,3,4])}
df = pandas.DataFrame(d)

pandas里没有直接指定索引的插入行方法，需自行设置
import pandas as pd
df = pd.DataFrame({'a': [1,2,3], 'b': ['a', 'b', 'c'], 'c': ["A", "B", "C"]})
# >>>a    b   c
# 0   1   a   A
# 1   2   b   B
# 2   3   c   C

line = pd.DataFrame({df.columns[0]: "--", df.columns[1]: "--", df.columns[2]: "--"}, index = [1])#取index为1的行，赋每列的值
df0 = pd.concat([df.loc[:0], line, df.loc[1: ]])
# >>>a    b   c
# 0   1   a   A
# 1  --  --   --
# 1   2   b   B
# 2   3   c   C

整个df数据框各列都可能有NaN,需把他替换成0，便于计算
from pandas import read_excel
df = pd.read_excel(r'', sheetname = 'Sheet3')
df.head()
#重！
df.replace('B', 'A')#use A replace B
df.replace({'English': 'cheat', 'Computer': 'absent' }, 0) #use 0 replace English columns's cheat people and Computer columns's absence
df.replace(['a', 'b'], ['A', 'B'])#use A&B replace a&b

直接使用df.reindex方法交换数据中的两行或两列
# >>>a    b   c
# 0   1   a   A
# 1   2   b   B
# 2   3   c   C
hang = [0, 2, 1]
df.reindex(hang)
# >>>a    b   c
# 0   1   a   A
# 2   3   c   C
# 1   2   b   B
lie = ['a', 'c', 'b']
df.reindex(columns = lie)

df.loc[[0,2], :] = df.loc[[2,0], :].values #exchange 0 and 2 row
df.loc[:, ['b', 'a']] = df.loc[:, ['a', 'b']].values #exchange two columns

Series的sort_index(ascending = True)#对index进行升序
sort_index(ascending = False) #降序
DataFrame中：df.sort_index(axis = 0, by = None, ascending = True)#by 针对某一列排序
排名方法：Series.rank(method = 'average', ascending = True)#method有average\min\max\first四个可选项
ser = Series([1,2,3,4], index = list('abcd'))

Series对象的reindex(index = None, **kwargs)#后者常用的参数有两个：method = None, fill_value = np.NaN

set_index() 重新设置某列为索引
DataFrame.set_index(keys,
                    drop = True,
                    append = False,
                    inplace = False)
append为True保留原索引加新索引
drop为False保留被作为索引的列
inplace为true在原数据集上修改

reset_index还原索引，使索引变为默认的整型索引
df.reset_index(level = None, drop = False, inplace = False, col_level = 0, col_fill='')

合并结构相同的两个数据框
pandas.concat([dataFrame1, dataFrame2,...])
df.append(df2, ignore_index = True) #df2追加到df上
pandas.concat([df1, df2], ignore_index = True) #后者表示index即可顺延

同一个数据框中不同列进行合并
X = x1 + x2 +...

不同结构的数据框按照一定的条件进行匹配合并
表一：姓名、学号。表二：学号，导师
merge(x, y, left_on, right_on)#后二参数表示两个表中用于匹配的列

离差标准化：min-max标准化
X* = (x - min)/(max - min)

Z-score标准化
X* = (x - μ)/σ

按一定的数据指标，把数据划分为不同的区间来进行研究
cut(series, bins, right = True, labels = NULL)
# series表示需分组的数据
# bins表分组的依据数据
# right表分组时右边是否闭合
# labels表分组自定义标签

将字符型的日期格式转换为日期格式数据
to_datetime(dateString, format)
format的格式：
%Y：年份
%m：月份
%d：日期
（H， M， S）
to_datetime(df.date, format = '%Y/%m/%d')

将日期型数据按格式转化为字符型数据
apply(lambda x : 处理逻辑)
datetime.strftime(x, format)

从日期格式里抽取需要的部分属性
data_dt.dt.property

df.head()
df.tail(3) #查看头尾部

df.index
df.columns
df.values#显示索引、列和底层的numpy数据

df.describe() #对数据快速统计汇总

df.T #对数据进行转置

df.sort_index(axis = 1, ascending = False)#按轴进行排序

df.sort_values(by = 'B') #按值进行排序

df.loc[dates[0]] #使用标签来获取一个交叉区域
df.loc[:, ['A', 'B']] #通过标签在多个轴上进行选择
df.loc['2020102': '2020104', ['A', 'B']] #标签切片
df.loc['2020102', ['A', 'B']] #对于返回的对象进行维度缩减
df.loc[dates[0], 'A'] #获取一个标量
df.at[dates[0], 'A'] #快速访问一个标量

df.iloc[3] #通过传递数值进行位置选择
df.iloc[3:5, 0:2] #通过数值进行切片
df.iloc[[1,2,4], [0,2]] #通过指定一个位置的列表进行位置选择
df.iloc[1:3,:] #对行进行行切片
df.iloc[:, 1:3] #对列进行列切片
df.iloc[1,1]
df.iat[1,1] #获取特定的值

df[df.A > 0] #使用一个单独列的值来选择数据
np.where(df > 0) #使用where操作来选择数据
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
df2[df2['E'].isin(['two', 'four'])]

df.at[dates[0], 'A'] = 0 #通过标签设置新的值
df.iat[0, 1] = 0 #通过位置设置新的值
df.loc[:, 'D'] = np.array([5] * len(df)) #通过numpy数组设置一组新值

describe#描述性统计分析函数
size #计数
sum()
mean() #平均值
var() #方差
std() #标准差
df.Math.describe()
out:
count
mean
std
min
25%
50%
75%
max

import numpy as np
np.mean(df['Math'])
np.average(df['Math'])

df['Math'].mean()

df.mode() #众数

常用：计数 求和 平均值
df.groupby(by = ['分类1', '分类2',...])['被统计的列'].agg({列别名1: 统计函数1, ...})
df.groupby('Class')['sports', 'Math', 'English'].mean()
df.groupby(by = ['Class', 'Sex'])['English'].agg(['Sum' : np.sum, 'average' : np.mean, ...])

df['sum'] = df.English + df.Chinese + df.Math
df['sum'].head()
df['sum'].describe()
bins = [min(df.sum) - 1, 400, 450, max(df.sum) + 1] #divided three segment
labels = ['under 400', 'between 400 and 450', 'over 450'] #label to them
总分分层 = pd.cut(df.sum, bins, labels = labels)
总分分层.head()
df['总分分层'] = 总分分层
df.tail()

pivot_table(values, index, columns, aggfunc, fill_value)
above:
数据透视表中的值
数据透视表中的行
数据透视表中的列
统计函数
NA值的统一替换
df.pivot_table(values = ['sum'], index = ['总分分层'],
               columns = ['sex'], aggfunc = [numpy.size, numpy.mean])

df_pt = df.pivot_table(values = ['sum'], index = ['总分分层'],
               columns = ['sex'], aggfunc = [numpy.size, numpy.mean])
df_pt.sum()
df_pt.sum(axis = 1) #按列合计
df_pt.div(df_pt.sum(axis = 1), axis = 0) #按列占比
df_pt.div(df_pt.sum(axis = 0), axis = 1) #按行占比

0 <= |r| < 0.3 低度相关
0.3 <= |r| < 0.8 中度相关
0.8 <= |r| <= 1 高度相关
相关分析函数：
DataFrame.corr()
Series.corr(other)

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#example data
mu = 100    #分布的均值
sigma = 15  #分布的标准差
x = mu + sigma * np.random.randn(10000)
print("x:", x.shape)
#直方图的条数
num_bins = 50
#绘制直方图
n, bins, patches = plt.hist(x, num_bins, normed = 1, facecolor = 'green', alpha = 0.5)
#添加一个最佳拟合和曲线
y = mlab.normpdf(bins, mu, sigma) #返回关于数据的pdf数值（概率密度函数）
plt.plot(bins, y, 'r--')
plt.xlabel('Smarts')
plt.ylabel('Probability')
#在途中添加公式需要的latex语法
plt.title('Histogram of IQ: $\mu = 100$, $\sigma = 15$')
#调整图像的间距，防止Y轴数值与label重合
plt.subplots_adjust(left = 0.15)
plt.show()
print("bind :\n", bins)

#matplotlib绘制三维图像
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#生成数据
delta = 0.2
x = np.arange(-3, 3, delta)
y = np.arange(-3, 3, delta)
X, Y = np.meshgrid(x, y)
Z = X ** 2 + Y ** 2
x = X.flatten()
#返回一维的数组，但该函数只能适用于numpy对象（array， mat）
y = Y.flatten()
z = Z.flatten()
fig = plt.figure(figsize = (12, 6))
ax1 = fig.add_subplot(121,projection = '3d')
ax1 = plot_trisurf(x, y, z, cmap = cm.jet, linewidth = 0.01)
#cmp指颜色， 默认绘制为RGB颜色空间，jet表示“蓝 青 黄 红”颜色
plt.title("3D")
ax2 = fig.add_subplot(122)
cs = ax2.contour(X, Y, Z, 15, cmap = 'jet', ) #15表示等高线的密集程度，正比
ax2.clabel(cs, inline = True, fontsize = 10, fmt = '%1.1f')
plt.title("Contour")
plt.show()

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure(figsize = (8, 6))
ax = fig.gca(projection = '3d')
#生成三维测试数据
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride = 8, cstride = 8, alpha = 0.3)
cset = ax.contour(X, Y, Z, zdir = 'z', offset = -100, cmap = cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir = 'x', offset = -40, cmap = cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir = 'y', offset = 40, cmap = cm.coolwarm)
ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)
plt.show()
help(ax.plot_surface) #了解具体函数的用法

import matplotlib.pyplot as plt
import pandas as pd
df_data = pd.read_csv('test.csv')
df_data.head()
#作图
fig, ax = plt.subplots()
#设置气泡颜色
colors = ["#99CC01"]
ax.scatter(df_data['SepalLength'], df_data['SepalWidth'], s = df_data['PetalLength'] * 100, color = colors, alpha = 0.6)
ax.set_xlabel('SpealLength(cm)')
ax.set_ylabel('SpealWidth(cm)')
ax.set_title('PetalLength(cm)* 100')
#显示网格
ax.grid(True)
fig.tight_layout()
plt.show()

from PIL import Image
#读取图片文件
pil_im = Image.open(r'test.jpg')
#转为灰度图
Pil_im = Image.open(r'test.jpg').convert("L")
pil_im

# thumbnail()接收一个元组参数（该参指定生成缩略图大小）
pil_im.thumbnail((128, 128))
pil_im

from PIL import Image
pil_im = Image.open(r'test.jpg')
box = (150, 350, 400, 600)
region = pil_im.crop(box)
region = region.transpose(Image.ROTATE_90)
pil_im.paste(region, box)
pil_im.show()

#读取文件
pil_im = Image.open(r'test.jpg')
out = pil_im.resize((128, 128))
out = out.rotate(45)


import numpy as np
import matplotlib.pyplot as plt
im = np.array(Image.open(r'test.jpg').convert("L"))
print("the size of picture", im.shap)
#图像轮廓
plt.figure()
#不使用颜色信息
plt.gray()
#在原点的左上角显示图像轮廓
plt.contour(im, origin = "image")
plt.axis("equal")
plt.show()
#直方图
plt.hist(im.flatten(), 128)
plt.show()

# opencv是一个c++库，用于实时处理计算机视觉问题
# anaconda下 conda install opencv
# 函数imread()返回一个标准的numpy数组，imwrite()会根据文件后缀自动转换图像
import cv2
#read image
im = cv2.imread(r'test.jpg')
print(im.shape)
#save image
cv2.imwrite(r'test.png', im)

# opencv中图像不是按传统的RGB存储的，而是BGR的顺序，颜色空间转换可以用cvtColor()来实现
im = cv2.imread(r'test.jpg')
#创建灰度图像
gray = cv2.cvtColor(im, cv2, COLOR_BGR2GRAY)
print(gray)
print(gray.shape)
#最有用的一些转换代码
cv2.COLOR_BGR2GRAY
cv2.COLOR_BGR2RGB
cv2.COLOR_GRAY2BGR

#图像显示，用matplotlib
import matplotlib.pyplot as plt
import cv2
#读取图片
im = cv2.imread(r'test.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#计算图像的积分
intim = cv2.integral(gray)
#归一化并保存
intim = (255 * intim) / intim.max()
plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1)
plt.imshow(gray)
plt.title('YTZ picture')
plt.subplot(1, 2, 2)
plt.imshow(intim)
plt.title('YTZ integral')
plt.show()

# str.capitalize()
# str.casefold()
# str.lower()
# str.upper()
# str.count(sub[,start[,end]])
# str.encode(encoding="utf-8", errors="strict")
# str.find(sub[,start[,end]])
# str.format(*args, **kwargs)
# str.join(iterable)
# str.strip([chars])
# str.lstrip([chars])
# str.rstrip([chars])
# str.replace(old, new[,count])
# str.split(sep = None, maxsplit=-1)
# "\d" 匹配一个数字
# "\w" 匹配一个字母或数字
# "." 匹配任意字符
# * 表示任意个字符（包括0个）
# + 至少一个字符
# ？表示0个或1个字符
# {n} 表示n个字符
# {n, m} n到m个
# eg:\d{3}\s+\d{3,8} 三个数字至少一个空格3到8个数字
# 转义：‘\’
# [] 表示范围
# [0-9a-zA-Z\_] 可以匹配一个数字、字母或者下划线
# [0-9a-zA-Z\_]+ 可以匹配至少由一个数字、字母或者下划线组成的字符
# [a-zA-Z\_][0-9a-zA-Z\_]* 可以匹配由字母或下划线开头，后接任意个由一个数字、字母、下划线组成的字符串（Python的合法变量）
# [a-zA-Z\_][0-9a-zA-Z\_]{0,19} 精确的限制了变量的长度是1~20个字符（前面一个+后面最多19个）
# A|B 可以匹配A或B (P|p)ython
# ^ 表示行的开头，^\d 表示必须以数字开头
# $ 表示行的结束，^d$ 表示必须以数字结束
# 注意：py也可匹配'python',但若加上^、$，即^py$，就变成整行匹配

# 使用python的r前缀，就不用考虑r的转义问题
# s = r'ABC\-001'
# Match()方法判断是否匹配，成功返回match对象，否则None
# test = '用户输入的字符串'
# if re.match(r'正则表达式', test):
#     print('ok')
# else:
#     print('failed')

# 正常的切分代码：
# 'a b   c'.split(' ')
# ['a', 'b', ' ', ' ', 'c']
# 正则表达式：
# re.split(r'\s+', 'a b   c')
# ['a', 'b', 'c']
# '\,' re.split(r'[\s+\,]+', 'a, b, c  d') 多少个空格都可以正常分割
# '\,\;' re.split(r'[\s\,\;]+', 'a, b;; c  d')

# ()表示分组
# m = re.match(r'^(\d{3})-(\d{3,8})$', '010-12345')
# m.group(0) 全部
# m.group(1) 前部
# m.group(2) 后部

# re.match(r'^(\d+)(0*)$', '102300').groups()
# ('102300', ' ') 由于\d+采用贪婪匹配，直接把102300全匹配了，导致后面空字符串
# 加个？就能让\d+采用非贪婪匹配
# re.match(r'^(\d+?)(0*)$', '102300').groups()
# ('1023', '00')

# 当一个正则式要重复使用几千次，可以采用预编译
# import re
# re_telephone = re.compile(r'^(\d{3})-(\d{3,8})$')
# re_telephone.match('010-12345').groups()
# re_telephone.match('010-8086').groups()

a = 'hello the world'
str_gb2312 = a.encode('gb2312')

gb2312_utf8 = str_gb2312.decode('gb2312').encode('utf-8')

utf8_gbk = gb2312_utf8.decode('utf-8').encode('gbk')

utf8_str = utf8_gbk.decode('gbk')

# 打开文件并读取
file = open(r'C:\test_utf8.txt')
f = file.read() #read each row #报错，默认或r读取二进制文件，可能出现文档读取不全
file = open(r'C:test_utf.txt', 'rb') #rb读取二进制文件
f = file.read()
print(f)
f1 = f.decode('utf-8')
print(f1)

# 通过python内置的urllib.request模块可以很轻松的获得网页的字节码，通过对字节码的解码就可以获取到网页的源码字符串
from urllib import request
fp = request.urlopen('http://www.baidu.com')
content = fp.read()
fp.close()
type(content)
html = content.decode()
# 并不是所有网页都采用utf-8编码方式，这时需要chardet库判断编码方式
import chardet
det = chardet.detect(content)
det
if det['confidence'] > 0.8:
    html = content.decode(det['encoding'])
    print(det['encoding'])
else:
    html = content.decode('gbk')
    print(det['encoding'])

# Beautiful Soup是一个可以从HTML或XML文件中提取数据的python库，从网页抓取数据，
# 它提供了一些简单的、python式的函数，处理导航、搜索、修改分析树等功能。它可以自动
# 将输入文档转换为Unicode编码，输出文档转换成utf-8，不需要考虑编码方式
# 安装库:pip install beautifulsoup4
from bs4 import BeautifulSoup
html = """
......."""
soup = BeautifulSoup(html)
print(soup.prettify())
for a in soup.findAll(name = 'a'):#找出所有a的标签
    print('attrs: ', a.attrs)
    print('string；', a.string)
    print('------------------------')

for tag in soup.findAll(attrs = {"class":"sister", "id":"link1"}):
# 找到所有的class是sister id是link1的标签
    print('tag: ', tag.name)
    print('attrs: ', tag.attrs)
    print('string: ', tag.string)
for tag in soup.findAll(name = 'a', text = "Elsie"):#所有内容包含Elsie的标签
    print('tag: ', tag.name)
    print('attrs: ', tag.attrs)
    print('string: ', tag.string)
import re
for a in soup.findAll('a', text = re.compile(".*?ie")):#找出所有结尾为ie的a的标签
    print(a)
def parser(tag):
    """
    自定义解析函数：解析出标签名为'a'，属性不为空且id属性为link1的标签
    """
    if tag.name == 'a' and tag.attrs and tag.attrs['id'] == 'link1':
        return True
for tag in soup.findAll(parser):
    print(tag)

# python自带的csv模块可以处理csv文件
csv = '''id,name, score
1.Lily, 23
2.Lory, 18
3.Dacy, 25'''
with open('G:/test.csv', 'w') as f:
    f.write(csv)

import sqlite3 as base
db = base.connect('d:/test.db')
'''数据库文件存在时，直接连接；不存在时，创建相应数据库文件。
此时当前目录下能找到对应的数据库文件'''
#获取游标
sur = db.cursor()
#建表
sur.execute("""create table info(
id text,
name text,
score text)""")
db.commit()
#添加数据
sur.execute("insert into info values('1', 'Lily', '23')")
sur.execute("insert into info values('2', 'Lory', '18')")
sur.execute("insert into info values('3', 'Dacy', '25')")
db.commit()
sur.close()
db.close()
#用工具软件SQLiteSpy打开

# 通过前面的学习，知道了怎么下载网页源码、解析网页、保存数据。
#引入包
from urllib import request
from chardet import detect
from bs4 import BeautifulSoup
#获取网页源码
def getSoup(url):
    with request.urlopen(url) as fp:
        byt = fp.read()
        det = detect(byt)
        return BeautifulSoup(byt.decode(det['encoding']), 'lxml')
#解析数据 F12找到标签的位置
def getData(soup):
    '''获取数据'''
    data = []
    ol = soup.find('ol', attrs = {'class': 'grid_view'})
    for li in ol.findAll('li'):
        tep = []
        titles = []
        for span in li.findAll('span'):
            if span.has_attr('class'):
                if span.attrs['class'][0] == 'title':
                    titles.append(span.string.strip())
                elif span.attrs['class'][0] == 'rating_num':
                    tep.append(span.string.strip())
                elif span.attrs['class'][0] == 'inq':
                    tep.append(span.string.strip())
                    tep.insert(0, titles)
                    data.append(tep)
                return data
#获取下一页的链接
def nextUrl(soup):
    '''获取下一页链接后缀'''
    a = soup.find('a', text = re.compile("^后页"))
    if a:
        return a.attrs['href']
    else:
        return None
#组织代码结构
if __name__ == '__main__':
    url = "https://movie.douban.com/top250"
    soup = getSoup(url)
    print(getData(soup))
    nt = nextUrl(soup)
    while nt:
        soup = getSoup(url + nt)
        print(getData(soup))
        nt = nextUrl(soup)

# Scrapy是Python开发的一个快速、高层次的屏幕抓取和web抓取框架，用于抓取web站点并从网页中提取结构化的数据
# 依赖于Twisted、lxml、pywin32等包，安装包之前安装vc++10.0，安装包的顺序是：pywin32\Twisted\lxml\Scrapy
# conda install pywin32
import os
pname = input('项目名：')
os.system("scrapy startproject " + pname)
os.chdir(pname)
wname = input('爬虫名：')
sit = input('网址：')
os.system('scrapy genspider ' + wname + ' ' + sit)
runc = """
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from %s.spiders.%s import %s
#获取settings.py模块的设置
settings = get_project_settings()
process = CrawlerProcess(settings = settings = settings)
#可以添加多个spider
process.craw1(Spider1)
process.craw1(Spider2)
process.craw1(%s)
#启动爬虫，会阻塞，直到爬取完成
process.start()""" % (pname, wname, wname[0].upper() + wname[1:] + 'Spider', wname[0].upper() + wname[1:] + 'Spider')
with open('main.py', 'w', encoding='utf-8') as f:
    f.write(runc)
input('end')
#运行上述文件初始化代码模块

# 进入项目结构，找到items.py并进行修改，确定需要爬取的项目，代码:
from scrapy import Item, Field
class DoubanItem(Item):
    name = Field()
    fen = Field()
    words = Field()
#进入项目结构,修改爬虫文件
import scrapy
from douban.items import DoubanItem
from bs4 import BeautifulSoup
import re
class Top250Spider(scrapy.Spider):
    name = 'top250'
    allowed_domains = ['movie.douban.com']
    start_urls = ['https://movie.douban.com/top250/']
    def parse(self, response):
        soup = BeautifulSoup(response.body.decode('utf-8', 'ignore'), 'lxml')
        ol = soup.find('ol', attrs = {'class': 'grid_view'})
        for li in ol.findAll('li'):
            tep = []
            titles = []
            for span in li.findAll('span'):
                if span.has_attr('class'):
                    if span.has_attrs['class'][0] == 'title':
                        titles.append(span.string.strip().replace(',', ','))
                    elif span.attrs['class'][0] == 'rating_num':
                        tep.append(span.string.strip().replace(',', ','))
                    elif span.attrs['class'][0] == 'inq':
                        tep.append(span.string.strip().replace(',', ','))
            tep.insert(0, titles[0])
            while len(tep) < 3:
                tep.append("-")
            tep = tep[:3]
            item = DoubanItem()
            item['name'] = tep[0]
            item['fen'] = tep[1]
            item['words'] = tep[2]
            yield item
        a = soup.find('a', text = re.compile("^后页"))
        if a:
            yield scrapy.Request("https://movie.douban.com/top250" + a.attrs['href'], callback = self.parse)
#以上在parse函数里将源码解码成soup对象,解析出数据item通过生成器yield返回,解析出接下来需要爬行的url通过request对象yield到
# 爬行队列,指定处理该url的处理函数为self.parse.
#修改数据存储文件,将爬取的数据导入csv文件
import csv
class DoubanPipeline(object):
    def __init__(self):
        self.fp = open('TOP250.csv', 'w', encoding='utf-8')
        self.wrt = csv.DictWriter(self.fp, ['name', 'fen', 'words'])
        self.wrt.writeheader()

    def __del__(self):
        self.fp.close()
    def process_item(self, item, spider):
        self.wrt.writerow(item)
        return item
#修改配置文件
BOT_NAME = 'douban'
SPIDER_MODULES = ['douban.spiders']
NEWSPIDER_MODULES = 'douban.spiders'
#豆瓣必须加这个
USER_AGENT = 'Mozilla/5.0(Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'
ROBOTSTXT_OBEY = False
ITEM_PIPELINES = {
    'douban.pipelines.DoubanPipeline':300,
}
#到此为止就可以运行主程序了

# jieba.cut(s, cut_all = True) s is string and cut_all是否采用全模式
import jieba
seg_list = jieba.cut("我来到清华大学", cut_all = True)
print("Full Mode:", "/ ".join(seg_list))
# result:我/来到/清华/清华大学/华大/大学
seg_list = jieba.cut("我来到清华大学", cut_all = False)#精确模式
print("Default Mode:", "/ ".join(seg_list))
result:我/来到/清华大学
# jieba.cut_for_search(s) s需要被分词的字符串，该方法适合用于搜索引擎构建倒排序索引的分词，粒度比较细
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print(",".join(seg_list))
# result:小明，硕士，毕业，于，中国，科学，学院，科学院，中国科学院，if 计算...


class Vector:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        return 'Vector(%d, %d)' % (self.a, self.b)
    def __add__(self, other):
        return Vector(self.a + other.a, self.b + other.b)
#生成器
import sys
def fibonacci(n):
    a, b, counter = 0, 1, 0
    while True:
        if (counter > n):
            return
        yield a
        a, b = b, a + b
        counter += 1
f = fibonacci(10)
while True:
    try:
        print(next(f), end=" ")
    except StopIteration:
        sys.exit()
#起始
def hcf(x, y):
    smaller = y if x > y else x
    for i in range(1, smaller + 1)
        if ((x % i == 0) and (y % i == 0)):
            hcf = i
    return hcf

def lcm(x, y):
    greater = x if x > y else y
    while True:
        if greater % x == 0 and greater % y == 0:
            lcm = greater
            break
        greater += 1
    return lcm

def add(x, y):
    return x + y
def subtract(x, y):
    return x - y
def multiply(x, y):
    return x * y
def divide(x, y):
    return x / y
print("Options:\n"
      "1.add\n"
      "2.subtract\n"
      "3.multiply\n"
      "4.divide\n")
choice = input("Please input your choose number")
if choice == '1':
    print()
elif choice == '2':
    print()
elif choice == '3':
    print()
elif choice == '4':
    print()
else:
    print("it is error")

import calendar
yy = int(input("Input year:"))
mm = int(input("Input month:"))
print(calendar.month(yy,mm))

def recur_fibo(n):
    if n <= 1:
        return n
    else:
        return(recur_fibo(n-1) + recur_fibo(n-2))
nterms = int(input('the number of term'))
if nterms <= 0:
    print('input a positive number')
else:
    print('fabnaaci')
    for i in range(nterms):
        print(recur_fibo(i))


with open("test.txt", "wt") as out_file:
    out_file.write("this sentence will  write in text")
with open("test.txt", "rt") as in_file:
    text = in_file.read()

print(text)

str = 'runoob'
print(str.isalnum())
print(str.isalpha())
print(str.isdigit())
print(str.islower())
print(str.isupper())

str = "runoob.com"
print(str.upper())
print(str.lower())
print(str.capitalize())
print(str.title())

import calendar
monthRange = calendar.monthrange(2016, 9)
print(monthRange)

import datetime
def getYesterday():
    today = datetime.date.today()
    oneday = datetime.timedelta(days=1)
    yesterday = today - oneday
    return yesterday
def getYesterday():
    yesterday = datetime.date.today() + datetime.timedelta(-1)
    return yesterday


params = {"a":1, "b":2, "c":3}
print(["%s = %s" % (k, v) for k, v in params.items()])
list = ";".join(["%s = %s" % (k, v) for k, v in params.items()])
print(list)
s = ",".join(list)
s.split(';')
s.split(',', 1)
li = [1,2,3,4]
print([elem * 2 for elem in li])
print([k for k in params.items()])
print([v for v in params.items()])
print(["%s = %s" % (k, v) for k, v in params.items()])
[elem for elem in li if len(elem) > 1]
[elem for elem in li if elem != "b"]
[elem for elem in li if li.count(elem) == 1]

#约瑟夫环
people = {}
for x in range(1, 31):
    people[x] = 1
check = 0
i = 1
j = 0
while i <= 31:
    if i == 31:
        i = 1
    elif j == 15:
        break
    else:
        if people[i] == 0:
            i += 1
            continue
        else:
            check += 1
            if check == 9:
                people[i] = 0
                check = 0
                print("{}号下船了".format(i))
                j += 1
            else:
                i += 1
                continue


def main():
    fish = 1
    while True:
        total, enough = fish, True
        for _ in range(5):
            if (total - 1) % 5 == 0:
                total = total - 1
            else:
                enough = False
                break
        if enough:
            print(f'总共有{fish}条鱼')
            break
        fish += 1
if __name__ == '__main__':
    main()


#实现秒表功能
import time
print("按下回车开始计时，按下 Ctrl + C 停止计时。")
while True:
    input("")
    starttime = time.time()
    print('begin')
    try:
        while True:
            print('计时: ', round(time.time() - starttime, 0), '秒', end="\r")
            time.sleep(1)
    except KeyboardInterrupt:
        print('over')
        endtime = time.time()
        print('总共的时间为:', round(endtime - starttime, 2),'secs')
        break

#翻转指定个数的元素
def leftRotate(arr, d, n): #数组长度为n, 翻转前面d个
    for i in range(d):
        leftRotatebyOne(arr, n)
def leftRotatebyOne(arr,n):
    temp = arr[0]
    for i in range(n-1):
        arr[i] = arr[i + 1]
    arr[n-1] = temp
def printArray(arr, size):
    for i in range(size):
        print("%d" % arr[i], end=" ")
arr = [1,2,3,4,5,6,7]
leftRotate(arr, 2, 7)
print(arr, 7)


def leftRotate(arr, d, n):
    for i in range(gcd(d, n)):
        temp = arr[i]
        j = i
        while 1:
            k = j + d
            if k >= n:
                k = k - n
            if k == i:
                break
            arr[j] = arr[k]
            j = k
        arr[j] = temp
def printArray(arr, size):
    for i in range(size):
        print ("%d" % arr[i], end=" ")

def gcd(a, b):
    if b == 0:
        return a;
    else:
        return gcd(b, a%b)

def printArray(arr, size):
    for i in range(size):
        print ("%d" % arr[i], end=" ")

def gcd(a, b):
    if b == 0:
        return a;
    else:
        return gcd(b, a%b)

def swapList(newList):
    newList[0], newList[-1] = newList[-1], newList[0]
    return newList

def swapList(list):
    get = list[-1], list[0]
    list[0], list[-1] = get
    return list

def swapPositions(list, pos1, pos2):
    first_ele = list.pop(pos1)
    second_ele = list.pop(pos2-1)
    list.insert(pos1, second_ele)
    list.insert(pos2, first_ele)

    return list

#列表翻转
def Reverse(lst):
    return [ele for ele in reversed(lst)]
def Reverse(lst):
    lst.reverse()
    return lst
def Reverse(lst):
    new_lst = lst[::-1]
    return new_lst

def clone_runoob(lst):
    lst_copy = lst[:]
    return lst_copy

def clone_one(lst):
    lst.copy = []
    lst.copy.extend(lst)
    return lst.copy
def clone_two(lst):
    list_copy = list(lst)
    return list_copy

def countX(list, x):
    return list.count(x)

total = 0
list1 = [1,2,3,4,5]
for ele in range(0, len(list1)):
    total = total + list1[ele]
ele = 0
while(ele < len(list1)):
    total = total + list1[ele]
    ele += 1

def sumOfList(list, size):
    if (size == 0):
        return 0
    else:
        return list[size - 1] + sumOfList(list, size - 1)
total = sumOfList(list1, len(list1))

def multiplyList(myList):
    resutl = 1
    for x in myList:
        result = result * x
    return result

from functools import reduce
list1 = [1,3,5,7]
sum = reduce(lambda x, y:x * y, list1)
print(sum)

def list_product(list_1, size):
    if size == 0:
        return 1
    else:
        return list_1[size-1] + list_product(list_1, size-1)
#求最小元素
list = [2,1,45,21]
list.sort()
min = list[:1]
print('the min number is ', *list[:1])

print(min(list))


test_str = "runoob"
new_str = test_str.replace(test_str[3], "", 1)

def ff(str, num):
    return str[:num] + str[num + 1:]

def check
