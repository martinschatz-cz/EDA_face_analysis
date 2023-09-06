#!/usr/bin/env python
# coding: utf-8

# # Subset relative point analysis - all points
# https://stackoverflow.com/questions/52859983/interactive-matplotlib-figures-in-google-colab

# In[1]:


get_ipython().system('pip install -r requirements.txt')


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# In[3]:


df = pd.read_csv("./data/cl5_frame_162.csv",sep=",")
df


# In[4]:


df0 = pd.read_csv("./data/cl5_frame_324.csv",sep=",")
df0


# In[5]:


df[["X","Y","Z"]].describe()


# In[6]:


sns.set(style = "darkgrid")

fig = plt.figure(figsize=(16, 12), dpi=80)
ax = fig.add_subplot(111, projection = '3d')

x = df['X']
y = df['Y']
z = df['Z']

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.scatter(x, y, z)
ax.view_init(-90, 90)

plt.show()


# In[7]:


stred = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]) #,1073,1083,1133,1134,1188,1231,1268,1209,1167])


# In[8]:


# Provide 'Address' as the column name
size = df.shape[0]
info = ["other" for x in range(size)]
df['info'] = info

for i in stred:
  df.loc[i,"info"]="center"

size = df.shape[0]
info = [0 for x in range(size)]
df['marker'] = info

for i in range((size)):
  df.loc[i,"marker"]=i


# In[9]:


size = df0.shape[0]
info = ["other" for x in range(size)]
df0['info'] = info

for i in stred:
  df0.loc[i,"info"]="center"

size = df0.shape[0]
info = [0 for x in range(size)]
df0['marker'] = info

for i in range((size)):
  df0.loc[i,"marker"]=i


# In[10]:


df0


# In[11]:


df.info()


# EyeLeft = 0
# LefteyeInnercorner = 210
# LefteyeOutercorner = 469
# LefteyeMidtop = 241
# LefteyeMidbottom = 1104
# RighteyeInnercorner = 843
# RighteyeOutercorner = 1117
# RighteyeMidtop = 731
# RighteyeMidbottom = 1090
# LefteyebrowInner = 346
# LefteyebrowOuter = 140
# LefteyebrowCenter = 222
# RighteyebrowInner = 803
# RighteyebrowOuter = 758
# RighteyebrowCenter = 849
# MouthLeftcorner = 91
# MouthRightcorner = 687
# MouthUpperlipMidtop = 19
# MouthUpperlipMidbottom = 1072
# MouthLowerlipMidtop = 10
# MouthLowerlipMidbottom = 8
# NoseTip = 18
# NoseBottom = 14
# NoseBottomleft = 156
# NoseBottomright = 783
# NoseTop = 24
# NoseTopleft = 151
# NoseTopright = 772
# ForeheadCenter = 28
# LeftcheekCenter = 412
# RightcheekCenter = 933
# Leftcheekbone = 458
# Rightcheekbone = 674
# ChinCenter = 4
# LowerjawLeftend = 1307
# LowerjawRightend = 1327

# In[12]:


points = pd.read_csv('points.csv', header=None, index_col=0, sep=";").squeeze("columns").to_dict()
points


# https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn791778(v=ieb.10)

# In[13]:


points_sel = np.fromiter(points.keys(), dtype=float)
# points_names = np.fromiter(points.values(), dtype=)


# In[14]:


# Provide 'Address' as the column name
size = df.shape[0]
info = ["other" for x in range(size)]
df['info'] = info

for i in points_sel:
  df.loc[i,"info"]="center"


# In[15]:


# Provide 'Address' as the column name
size = df0.shape[0]
info = ["other" for x in range(size)]
df0['info'] = info

for i in points_sel:
  df0.loc[i,"info"]="center"


# In[16]:


sns.set(style = "darkgrid")

fig = plt.figure(figsize=(16, 12), dpi=80)
ax = fig.add_subplot(111, projection = '3d')


names = pd.unique(df['info'])

df1=df[df['info'] == 'other']

x = df1['X']
y = df1['Y']
z = df1['Z']

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


ax.scatter(x, y, z, color = "green")

dfA=df[df['info'] == 'center']

x = dfA['X']
y = dfA['Y']
z = dfA['Z']

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.scatter(x, y, z, color = "red")

ax.view_init(-90, 90)

plt.show()


# In[17]:


sns.set(style = "darkgrid")

fig = plt.figure(figsize=(16, 12), dpi=80)
ax = fig.add_subplot(111, projection = '3d')


names = pd.unique(df0['info'])

df1=df0[df0['info'] == 'other']

x = df1['X']
y = df1['Y']
z = df1['Z']

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


ax.scatter(x, y, z, color = "green")

dfB=df0[df0['info'] == 'center']

x = dfB['X']
y = dfB['Y']
z = dfB['Z']

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.scatter(x, y, z, color = "red")

ax.view_init(-90, 90)

plt.show()


# In[18]:


dfA


# In[19]:


from scipy.spatial.distance import squareform, pdist

# pd.DataFrame(squareform(pdist(df.iloc[:, 1:3])), columns=df.marker.unique(), index=df.marker.unique())
exp_dist = pd.DataFrame(squareform(pdist(df.iloc[:, 1:3])), columns=df.marker.unique(), index=df.marker.unique())
exp_dist0 = pd.DataFrame(squareform(pdist(df0.iloc[:, 1:3])), columns=df0.marker.unique(), index=df0.marker.unique())


# In[20]:


diff = ((exp_dist-exp_dist0)/exp_dist).abs()


# In[21]:


diff.describe()


# In[22]:


diff.head()


# In[23]:


# # assume this df and that we are looking for 'abc'
# df = pd.DataFrame({'col':['abc', 'def','wert',1], 'col2':['asdf', 'abc', 'sdfg', 'def']})

# def get_rc(df,val):
#     return [(df[col][df[col]==val].index[i], df.columns.get_loc(col)) for col in df.columns for i in range(len(df[col][df[col]==val].index))]

# get_rc(diff,0.003228)


# In[24]:


keys = list(points.keys())
# get values in the same order as keys, and parse percentage values
vals = [str(points[k][:-1]) for k in keys]
# sns.barplot(x=keys, y=vals)


# In[25]:


# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(diff, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
cmap = sns.color_palette("Spectral", as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(diff, mask=mask, cmap=cmap, vmax=.01, center=0,
            square=True, linewidths=.0, cbar_kws={"shrink": .3})

# # set xticks labels
# ax.set(xticklabels=vals)
# # rotate labels
# plt.xticks(rotation=45)


# In[26]:


series = pd.Series(diff.values.ravel())
series.describe()


# In[27]:


mask = np.triu(np.ones_like(diff, dtype=bool))


# In[28]:


# 99 percentile
diff_array = diff.to_numpy()
res = list(zip(*np.where(diff_array >= series.quantile(0.99))))

print("99th Quantile: {}".format(series.quantile(0.99)))

# for i in range(len(res)):
#     print(points[keys[res[i][0]]] + ", " + points[keys[res[i][1]]])


# In[29]:


sns.set_theme(style="ticks")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(25, 6))
# ax.set_xscale("log")

keys = list(points.keys())
# get values in the same order as keys, and parse percentage values
vals = [str(points[k][:-1]) for k in keys]
# sns.barplot(x=keys, y=vals)

# Plot the orbital period with horizontal boxes
sns.boxplot(data=diff,
            whis=[0, 100], width=.6, palette="vlag")

# set xticks labels
# ax.set(xticklabels=vals)
# rotate labels
plt.xticks(rotation=45)

# # Add in points to show each observation
sns.stripplot(data=diff,
              size=4, palette='dark:.3', linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")

ax.set_title('Points relative % change overview')
ax.set_xlabel('Point')
ax.set_ylabel('Change in points')

sns.despine(trim=True, left=True)


# In[30]:


# a = diff.max()
a = diff[18]
# a.describe()
a.head()


# In[31]:


ma = a >= a.quantile(0.8)
ma


# In[32]:


sns.set(style = "darkgrid")

fig = plt.figure(figsize=(16, 12), dpi=80)
ax = fig.add_subplot(111, projection = '3d')


df1=df[ma == False]

x = df1['X']
y = df1['Y']
z = df1['Z']

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


ax.scatter(x, y, z, color = "blue")

# # under the threshold
# df1=df[ma == False]

# x = df1['X']
# y = df1['Y']
# z = df1['Z']

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")


# ax.scatter(x, y, z, color = "green")

# over the threshold
df2=df[ma == True]

x = df2['X']
y = df2['Y']
z = df2['Z']

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.scatter(x, y, z, color = "red", marker="h", s=30)

ax.view_init(-90, 90)

ax.set_title('Red points > 80 percentile of change')

plt.show()


# https://www.tutorialspoint.com/plot-scatter-points-on-a-3d-projection-with-varying-marker-size-in-matplotlib

# In[33]:


import numpy as np
from matplotlib import pyplot as plt
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

# df1=df[df['info'] == 'other']

# x = df1['X']
# y = df1['Y']
# z = df1['Z']

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# s = np.ones(df1.shape[0])

fig = plt.figure(figsize=(16, 12), dpi=80)
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, s=s, c=s, cmap='copper')

# under the threshold
df1=df

x = df1['X']
y = df1['Y']
z = df1['Z']

a_val=(a.to_numpy()*100)**2
a_val = np.nan_to_num(a_val, nan=1)
s=np.ceil(a_val)

# fig = plt.figure(figsize=(16, 12), dpi=80)
# ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=s, c=s, cmap='vlag')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(-90, 90)
plt.show()


# In[34]:


get_ipython().system('pip install watermark')


# In[35]:


from watermark import watermark
watermark(iversions=True, globals_=globals())
print(watermark())


# In[36]:


print(watermark(packages="watermark,numpy,pandas,seaborn"))


# In[ ]:




