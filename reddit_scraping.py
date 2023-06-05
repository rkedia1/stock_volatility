#!/usr/bin/env python
# coding: utf-8

# In[34]:


import praw

# Initialize Reddit instance
client_id = '_LYqCAq2rTb85eEHMeiOzQ'
client_secret = "emQOKlPN7y0aeRFwZFUBKB-WmUAsnw"
user_agent = "Stock Sentiment API"
username = "shudderisland098"
password = "sjikpnex973"

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    username=username,
    password=password
)


# In[35]:


sentiment_data = []

# Set subreddit
subreddit = reddit.subreddit("stocks")


# In[36]:


# Retrieve posts from all topics
posts = subreddit.search(query="stock", sort="new", time_filter="month")

for post in posts:
    sentiment_data.append(post.title)


# In[37]:


print(sentiment_data)


# In[38]:


get_ipython().run_line_magic('run', '__main__.py')


# In[47]:


get_ipython().system('pip list --outdated')


# In[42]:


#import __main__
from data import EquityData

equity_data = EquityData() 
data = equity_data.stock_data() 


# In[43]:


print(data)


# In[ ]:




