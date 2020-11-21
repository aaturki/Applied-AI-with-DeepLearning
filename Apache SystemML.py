#!/usr/bin/env python
# coding: utf-8

# # Assignment 4
# ## Understaning scaling of linear algebra operations on Apache Spark using Apache SystemML
# 
# In this assignment we want you to understand how to scale linear algebra operations from a single machine to multiple machines, memory and CPU cores using Apache SystemML. Therefore we want you to understand how to migrate from a numpy program to a SystemML DML program. Don't worry. We will give you a lot of hints. Finally, you won't need this knowledge anyways if you are sticking to Keras only, but once you go beyond that point you'll be happy to see what's going on behind the scenes.
# 
# So the first thing we need to ensure is that we are on the latest version of SystemML, which is 1.2.0:
# 
# The steps are:
# - pip install
# - start execution at the cell with the version - check

# In[1]:


from IPython.display import Markdown, display
def printmd(string):
    display(Markdown('# <span style="color:red">'+string+'</span>'))


if ('sc' in locals() or 'sc' in globals()):
    printmd('<<<<<!!!!! It seems that you are running in a IBM Watson Studio Apache Spark Notebook. Please run it in an IBM Watson Studio Default Runtime (without Apache Spark) !!!!!>>>>>')
    


# In[2]:


get_ipython().system('pip install pyspark==2.4.5')


# In[3]:


get_ipython().system('pip install systemml')


# In[4]:


from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
from pyspark.sql import SparkSession
spark = SparkSession     .builder     .getOrCreate()


# In[5]:


get_ipython().system('mkdir -p /home/dsxuser/work/systemml')


# In[6]:


from systemml import MLContext, dml
import numpy as np
import time
ml = MLContext(spark)
ml.setConfigProperty("sysml.localtmpdir", "mkdir /home/dsxuser/work/systemml")
print(ml.version())
    
if not ml.version() == '1.2.0':
    raise ValueError('please upgrade to SystemML 1.2.0, or restart your Kernel (Kernel->Restart & Clear Output)')


# Congratulations, if you see version 1.2.0, please continue with the notebook...

# We use an MLContext to interface with Apache SystemML. Note that we passed a SparkSession object as parameter so SystemML now knows how to talk to the Apache Spark cluster

# Now we create some large random matrices to have numpy and SystemML crunch on it

# In[7]:


u = np.random.rand(1000,10000)
s = np.random.rand(10000,1000)
w = np.random.rand(1000,1000)


# Now we implement a short one-liner to define a very simple linear algebra operation
# 
# In case you are unfamiliar with matrxi-matrix multiplication: https://en.wikipedia.org/wiki/Matrix_multiplication
# 
# sum(U' * (W . (U * S)))
# 
# 
# | Legend        |            |   
# | ------------- |-------------| 
# | '      | transpose of a matrix | 
# | * | matrix-matrix multiplication      |  
# | . | scalar multiplication      |   
# 
# 

# In[8]:


start = time.time()
res = np.sum(u.T.dot(w * u.dot(s)))
print (time.time()-start)


# As you can see this executes perfectly fine. Note that this is even a very efficient execution because numpy uses a C/C++ backend which is known for it's performance. But what happens if U, S or W get such big that the available main memory cannot cope with it? Let's give it a try:

# In[9]:


#u = np.random.rand(10000,100000)
#s = np.random.rand(100000,10000)
#w = np.random.rand(10000,10000)


# After a short while you should see a memory error. This is because the operating system process was not able to allocate enough memory for storing the numpy array on the heap. Now it's time to re-implement the very same operations as DML in SystemML, and this is your task. Just replace all ###your_code_goes_here sections with proper code, please consider the following table which contains all DML syntax you need:
# 
# | Syntax        |            |   
# | ------------- |-------------| 
# | t(M)      | transpose of a matrix, where M is the matrix | 
# | %*% | matrix-matrix multiplication      |  
# | * | scalar multiplication      |   
# 
# ## Task

# In order to show you the advantage of SystemML over numpy we've blown up the sizes of the matrices. Unfortunately, on a 1-2 worker Spark cluster it takes quite some time to complete. Therefore we've stripped down the example to smaller matrices below, but we've kept the code, just in case you are curious to check it out. But you might want to use some more workers which you easily can configure in the environment settings of the project within Watson Studio. Just be aware that you're currently limited to free 50 capacity unit hours per month wich are consumed by the additional workers.

# In[10]:


script = """
U = rand(rows=1000,cols=10000)
S = rand(rows=10000,cols=1000)
W = rand(rows=1000,cols=1000)
res = sum( t(U) %*% (W * (U %*% S)))
"""


# To get consistent results we switch from a random matrix initialization to something deterministic

# In[11]:


prog = dml(script).output('res')
res = ml.execute(prog).get('res')
print(res)


# If everything runs fine you should get *6252492444241.075* as result (or something in that bullpark). Feel free to submit your DML script to the grader now!
# 
# ### Submission

# In[12]:


get_ipython().system('rm -f rklib.py')
get_ipython().system('wget https://raw.githubusercontent.com/romeokienzler/developerWorks/master/coursera/ai/rklib.py')


# In[13]:


from rklib import submit
key = "esRk7vn-Eeej-BLTuYzd0g"


email = "turki32235@gmail.com"


# In[14]:


part = "fUxc8"
token = "nQNm0YAQHWcn1nId" #you can obtain it from the grader page on Coursera (have a look here if you need more information on how to obtain the token https://youtu.be/GcDo0Rwe06U?t=276)
submit(email, token, key, part, [part], script)


# In[ ]:




