#!/usr/bin/env python
# coding: utf-8

# # Arkhamdb Investigator Popularity Racing Bar Chart
# Created: 2021-11-02  
# Updated: 2021-11-04  
# Author: Spencer Simon

# ## Overview

# This notebook uses data downloaded from [arkhamdb.com](https://arkhamdb.com) using their [public api](https://arkhamdb.com/api/).  The data downloading and cleaning is performed in the arkhamdb-data repository. 
# 
# This data is transformed and used to create a racing bar chart of investigator popularity over time in this notebook.  

# ## Setup

# ### Install and import libraries

# In[36]:


#import sys
#!{sys.executable} -m pip install git+https://github.com/programiz/bar_chart_race.git@master --user


# In[37]:


import pandas as pd
from datetime import datetime
import bar_chart_race as bcr
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


pd.set_option('display.max_columns', 50)


# ### Load Data

# In[3]:


df = pd.read_csv('../arkhamdb-data/investigator_popularity_raw.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# ## Data Preparation

# ### Drop duplicate decks

# In[6]:


df_temp = df.drop_duplicates(
    subset = ['name', 'date_creation'],
    keep = 'last').reset_index(drop = True)


# In[7]:


df_temp.shape


# In[8]:


print(f"Number of duplicate decks dropped: {df.shape[0] - df_temp.shape[0]}")


# ### Create date only field (without time), and date2 with only year-month

# In[9]:


df_temp['date'] = pd.to_datetime(df_temp['date_creation']).dt.date


# In[10]:


df_temp['date2'] = pd.to_datetime(df_temp['date_creation']).dt.date.apply(lambda x: x.strftime('%Y-%m'))


# In[11]:


df_temp.head()


# ### Drop all fields except investigator name and date

# In[12]:


df_small = df_temp[["date", "investigator_name"]]
df_small2 = df_temp[["date2", "investigator_name"]]


# In[13]:


df_small.head()


# ### Get count of decks built per investigator per day

# In[14]:


counts_series = df_small.groupby(['date', 'investigator_name']).size()
df_counts = counts_series.to_frame(name = 'count').reset_index()

counts_series2 = df_small2.groupby(['date2', 'investigator_name']).size()
df_counts2 = counts_series2.to_frame(name = 'count').reset_index()


# In[15]:


#df_counts.tail()
df_counts2.tail()


# ### Pivot df to wide format

# In[16]:


df_wide = pd.pivot_table(df_counts, values='count', index = 'date', 
                         columns = ['investigator_name'], aggfunc=np.sum, fill_value=0)

df_wide2 = pd.pivot_table(df_counts2, values='count', index = 'date2', 
                         columns = ['investigator_name'], aggfunc=np.sum, fill_value=0)


# In[17]:


df_wide2*12


# ### Add in rows for missing dates

# In[18]:


# New index range is min to max date
new_index = pd.date_range(df_wide.index.min(), df_wide.index.max()) 
df_wide = df_wide.reindex(new_index, fill_value=0)


# In[19]:


df_wide


# ### Calculate rolling sums
# For 1st df, Use 30 days as an approximation for the month and 90 days for 3 months  
# For df2, get 3 month and 6 month sums

# In[20]:


df_sums_30d = df_wide.copy()
cols=list(df_wide.columns)

df_sums_30d[pd.Index(cols)] = df_wide[cols].rolling(window=30).sum()


# In[21]:


df_sums_3m = df_wide2.copy()
cols=list(df_wide2.columns)

df_sums_3m[pd.Index(cols)] = df_wide2[cols].rolling(window=3).sum()


# In[22]:


# Calculate 90 day sums
df_sums_90d = df_wide.copy()
df_sums_90d[pd.Index(cols)] = df_wide[cols].rolling(window=90).sum()


# In[23]:


# Calculate 6 month sums
df_sums_6m = df_wide2.copy()
df_sums_6m[pd.Index(cols)] = df_wide2[cols].rolling(window=6, min_periods=3).sum()


# In[24]:


df_sums_6m[2:]*2


# ## Initial Visualizations

# ### Time series of number of decks created
# Based on https://www.python-graph-gallery.com/web-line-chart-with-labels-at-line-end

# In[25]:


df_ts = df_counts2.copy()


# In[26]:


df_ts['date'] = pd.to_datetime(df_ts['date2'])


# In[27]:


df_ts.head()


# In[28]:


# A list of investigators that are going to be highlighted
HIGHLIGHTS = ["Zoey Samaras", "Roland Banks", "Jenny Barnes"]


# In[29]:


# Create 'group' to determine which ones are highlighted
df_ts["group"] = np.where(
    df_ts["investigator_name"].isin(HIGHLIGHTS),
    df_ts["investigator_name"],
    "other"
)

# Make 'group' categorical 
df_ts["group"] = pd.Categorical(
    df_ts["group"], 
    ordered=True,  
    categories=sorted(HIGHLIGHTS) + ["other"]
)


# In[30]:


# Shades of gray
GREY10 = "#1a1a1a"
GREY30 = "#4d4d4d"
GREY40 = "#666666"
GREY50 = "#7f7f7f"
GREY60 = "#999999"
GREY75 = "#bfbfbf"
GREY91 = "#e8e8e8"
GREY98 = "#fafafa"

# Colors used to shade countries
COLOR_SCALE = [
    "#7F3C8D", # ARG
    "#11A579", # BRA
    "#3969AC", # CHE
    "#F2B701", # DNK
    "#E73F74", # EUZ
    "#80BA5A", # GBR
    "#E68310", # SWE
    GREY50     # USA
]

# Vertical lines every 5 years
VLINES = np.arange(2016, 2021, 1)


# In[31]:


# Initialize layout ----------------------------------------------
fig, ax = plt.subplots(figsize = (14, 8.5))

# Background color
fig.patch.set_facecolor(GREY98)
ax.set_facecolor(GREY98)

# Vertical lines used as scale reference
#for h in VLINES:
#    ax.axvline(h, color=GREY91, lw=0.6, zorder=0)

# Horizontal lines
ax.hlines(y=np.arange(-4, 4), xmin=2016, xmax=2021, color=GREY91, lw=0.6)

# Darker horizontal line at y=0
ax.hlines(y=0, xmin=2016, xmax=2021, color=GREY60, lw=0.8)

# Vertical like at x = 2008
#ax.axvline(2018, color=GREY40, ls="dotted")

# Annotations indicating the meaning of the vertical line
#ax.text(2018.15, -3.35, "2018", fontname="Montserrat", 
#        fontsize=14, fontweight=500, color=GREY40, ha="left")

# Add lines ------------------------------------------------------
# Create one data frame for the highlighted countries, and other
# for non-highlighted countries.
df_highlight = df_ts[df_ts["group"] != "other"]
df_others = df_ts[df_ts["group"] == "other"]

for group in df_others["investigator_name"].unique():
    data = df_others[df_others["investigator_name"] == group]
    ax.plot("date", "count", c=GREY75, lw=1.2, alpha=0.5, data=data)

for idx, group in enumerate(df_highlight["investigator_name"].unique()):
    data = df_highlight[df_highlight["investigator_name"] == group]
    color = COLOR_SCALE[idx]
    ax.plot("date", "count", color=color, lw=1.8, data=data)
    
ax.set_xlim([datetime(2016, 9, 1), datetime(2021, 10, 1)])


# In[32]:


#datetime.date(2014, 1, 26)


# ## Create Racing Bar Chart

# From https://github.com/andresberejnoi/bar_chart_race/tree/image_labels
# Updates:
# - Make df with only the rolling sums
# - Make col names of this new df just investigator names
# - decide on if 30 day sums are best measure
# - Make bcr with options: images, top X bars, colors, etc.
# - continue cropping w/ Leo

# In[38]:


# Test with monthly data * 12
# This is in Python with new bcr version; not working 
bcr.bar_chart_race(
    df=df_wide2*12,
    img_label_folder='images',
    filename='test.mp4',
    orientation='h',
    sort='desc',
    n_bars=10,
    fixed_order=False,
    #fixed_max=True,
    steps_per_period=45,
    interpolate_period=True,
    #label_bars=True,
    bar_size=.95,
    #period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},
    #period_fmt='%B %d, %Y',
    #period_summary_func=lambda v, r: {'x': .99, 'y': .18,
    #                                  's': f'Total deaths: {v.nlargest(6).sum():,.0f}',
    #                                  'ha': 'right', 'size': 8, 'family': 'Courier New'},
    #perpendicular_bar_func='median',
    period_length=1500,
    fig_kwargs={'figsize': (5, 3), 'dpi': 144},
    colors='dark12',
    title='Investigator Popularity',
    #title_size='',
    bar_label_font=7,
    tick_label_font=7,
    shared_fontdict={'family' : 'Helvetica', 'color' : '.1'},
    scale='linear',
    writer=None,
    fig=None,
    bar_kwargs={'alpha': .7},
    filter_column_colors=False)

# In[34]:

# works with bcr version in Jupyter
# Test with monthly (6 month) data multiplied by 2 (decks per year)
bcr.bar_chart_race(
    df=df_sums_6m[2:]*2,
    filename=None,
    orientation='h',
    sort='desc',
    n_bars=10,
    fixed_order=False,
    #fixed_max=True,
    steps_per_period=30,
    interpolate_period=False,
    label_bars=True,
    bar_size=.95,
    #period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},
    #period_fmt='%B %d, %Y',
    #period_summary_func=lambda v, r: {'x': .99, 'y': .18,
    #                                  's': f'Total deaths: {v.nlargest(6).sum():,.0f}',
    #                                  'ha': 'right', 'size': 8, 'family': 'Courier New'},
    #perpendicular_bar_func='median',
    period_length=1000,
    figsize=(5, 3),
    dpi=144,
    cmap='dark12',
    title='Investigator Popularity',
    title_size='',
    bar_label_size=7,
    tick_label_size=7,
    shared_fontdict={'family' : 'Helvetica', 'color' : '.1'},
    scale='linear',
    writer=None,
    fig=None,
    bar_kwargs={'alpha': .7},
    filter_column_colors=False)


# In[ ]:


# Do like above, with 1 month numbers * 12
# Try Percentages


# In[ ]:





# In[ ]:


# Example from documentation

bcr.bar_chart_race(
    df=df_wide,
    filename='covid19_horiz.mp4',
    orientation='h',
    sort='desc',
    n_bars=6,
    fixed_order=False,
    fixed_max=True,
    steps_per_period=10,
    interpolate_period=False,
    label_bars=True,
    bar_size=.95,
    period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},
    period_fmt='%B %d, %Y',
    period_summary_func=lambda v, r: {'x': .99, 'y': .18,
                                      's': f'Total deaths: {v.nlargest(6).sum():,.0f}',
                                      'ha': 'right', 'size': 8, 'family': 'Courier New'},
    perpendicular_bar_func='median',
    period_length=500,
    figsize=(5, 3),
    dpi=144,
    cmap='dark12',
    title='COVID-19 Deaths by Country',
    title_size='',
    bar_label_size=7,
    tick_label_size=7,
    shared_fontdict={'family' : 'Helvetica', 'color' : '.1'},
    scale='linear',
    writer=None,
    fig=None,
    bar_kwargs={'alpha': .7},
    filter_column_colors=False)

