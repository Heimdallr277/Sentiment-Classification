
from __future__ import division
import os, sys

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import pandas as pd
inputfile = 'data/huizong.csv'
outputfile = 'data/haier_jd.txt'
data = pd.read_csv(inputfile, encoding='utf-8')
data = data[['评论']][data['品牌']=='海尔']
data.to_csv(outputfile, index=False, header=False, encoding='utf-8')


# In[6]:


inputfile='data/haier_jd.txt'
outputfile='data/haier_nomul.txt'
data=pd.read_csv(inputfile,encoding='utf-8',header=None)
l1=len(data)
data=pd.DataFrame(data[0].unique())
l2=len(data)
data.to_csv(outputfile,index=False,header=False,encoding='utf-8')
print('原始数据%s条评论，经过去重预处理后，删除了%s条评论，剩余%s条评论'%(l1,(l1-l2),l2))


# In[30]:


import csv, codecs

inputfile = 'data/haier/haier_nomul.txt'
outputfile_pos = 'data/haier/haier_jd_pos_1.txt'
outputfile_neg = 'data/haier/haier_jd_neg_1.txt'

outf_pos = codecs.open(outputfile_pos, 'wb',encoding='utf-8')
outf_neg = codecs.open(outputfile_neg, 'wb',encoding='utf-8')

with codecs.open(inputfile, 'r', encoding='utf-8') as inf:
    for line in inf:
        if not line.strip():
            continue
        sentscore = round((SnowNLP(line.strip()).sentiments - 0.5)*20)
        if sentscore > 0.0:
            outf_pos.write(str(sentscore)+'\t'+line.strip()+'\n')
        elif sentscore < 0.0:
            outf_neg.write(str(sentscore)+'\t'+line.strip()+'\n')

outf_pos.close()
outf_neg.close()


# In[34]:


inputfile1='data/haier_jd_pos_1.txt'
inputfile2='data/haier_jd_neg_1.txt'
outputfile1='data/haier_jd_pos_3.txt'
outputfile2='data/haier_jd_neg_3.txt'

data1=pd.read_csv(inputfile1,encoding='utf-8',header=None,delimiter="\n")
data2=pd.read_csv(inputfile2,encoding='utf-8',header=None,delimiter="\n")

data1=pd.DataFrame(data1[0].str.replace('.*?\d+?\\t ',''))
data2=pd.DataFrame(data2[0].str.replace('.*?\d+?\\t ',''))

data1.to_csv(outputfile1,index=False,header=False,encoding='utf-8')
data2.to_csv(outputfile2,index=False,header=False,encoding='utf-8')

l1=len(data1)
l2=len(data2)

print('正向评论有%s条'%l1)
print('负向评论有%s条'%l2)


# In[35]:


inputfile1='data/haier_jd_pos_1.txt'
inputfile2='data/haier_jd_neg_1.txt'
outputfile1='data/haier_jd_pos_3.txt'
outputfile2='data/haier_jd_neg_3.txt'

data1=pd.read_csv(inputfile1,encoding='utf-8',header=None,delimiter="\n")
data2=pd.read_csv(inputfile2,encoding='utf-8',header=None,delimiter="\n")

print(data1[0])
data1=pd.DataFrame(data1[0].str.replace('.*?\d+?\\t ',''))
data2=pd.DataFrame(data2[0].str.replace('.*?\d+?\\t ',''))

data1.to_csv(outputfile1,index=False,header=False,encoding='utf-8')
data2.to_csv(outputfile2,index=False,header=False,encoding='utf-8')

l1=len(data1)
l2=len(data2)

print('正向评论有%s条'%l1)
print('负向评论有%s条'%l2)


# In[49]:


inputfile1=codecs.open('data/haier_jd_pos_1.txt',mode='r',encoding='utf-8')
inputfile2=codecs.open('data/haier_jd_neg_1.txt',mode='r',encoding='utf-8')
outputfile1=codecs.open('data/haier_jd_pos_2.txt','wb',encoding='utf-8')
outputfile2=codecs.open('data/haier_jd_neg_2.txt','wb',encoding='utf-8')

line1=inputfile1.readline()
list1=[]
while line1:
    a1=line1.split()
    b1=a1[1:]
    list1.append(b1)
    line1=inputfile1.readline()
inputfile1.close()
#print(len(list1))
for i1 in list1:   
    outputfile1.write(''.join(i1)+'\n')
    
line2=inputfile2.readline()
list2=[]
while line2:
    a2=line2.split()
    b2=a2[1:]
    list2.append(b2)
    line2=inputfile2.readline()
inputfile2.close()
for i2 in list2:   
    outputfile2.write(''.join(i2)+'\n')

print('正向评论有%s条'%len(list1))
print('负向评论有%s条'%len(list2))


# In[66]:


import jieba.posseg as pseg
inputfile1='data/haier_jd_pos_2.txt'
inputfile2='data/haier_jd_neg_2.txt'
outputfile1=open('data/haier_jd_pos_cut.txt','w')
outputfile2=open('data/haier_jd_neg_cut.txt','w')

data1=pd.read_csv(inputfile1,encoding='utf-8',header=None)
data2=pd.read_csv(inputfile2,encoding='utf-8',header=None)

data_list1=[]
for i in range(len(data1[0])):
    word_cut1=pseg.cut(data1[0][i])
    line0=[]
    for w1 in word_cut1:
        if 'x'!=w1.flag:
            line0.append(w1.word)
    data_list1.append(' '.join(line0))
outputfile1.write('\n'.join(data_list1))

data_list2=[]
for i2 in range(len(data2[0])):
    word_cut2=pseg.cut(data2[0][i2])
    line1=[]
    for w2 in word_cut2:
        if 'x'!=w2.flag:
            line1.append(w2.word)
    data_list2.append(' '.join(line1))
outputfile2.write('\n'.join(data_list2))


# In[70]:


posfile='data/haier_jd_pos_cut.txt'
negfile='data/haier_jd_neg_cut.txt'
stoplist='data/stoplist.txt'

pos=pd.read_csv(posfile,header=None,engine='python')
neg=pd.read_csv(negfile,header=None,engine='python')
#csv格式默认以半角逗号为分割词，但该词在停用词表中，因而设置一个不存在的分割词比如tipdm，防止读取错误
stop=pd.read_csv(stoplist,encoding='utf-8',header=None,engine='python',sep='tipdm')
stop=[' ','']+list(stop[0])#pandas自动过滤空格，手动添加空格符

neg[1]=neg[0].apply(lambda s:s.split(' '))#定义分割函数
neg[2]=neg[1].apply(lambda x:[i for i in x if x not in stop])#依次判断是否是停用词
pos[1]=pos[0].apply(lambda s:s.split(' '))#定义分割函数
pos[2]=pos[1].apply(lambda x:[i for i in x if x not in stop])

from gensim import corpora,models

neg_dict=corpora.Dictionary(neg[2])#建立负面词典
neg_corpus=[neg_dict.doc2bow(i) for i in neg[2]]#建立语料库
neg_lda=models.LdaModel(neg_corpus,num_topics=3,id2word=neg_dict)#LDA模型训练

pos_dict=corpora.Dictionary(pos[2])#建立负面词典
pos_corpus=[pos_dict.doc2bow(i) for i in pos[2]]#建立语料库
pos_lda=models.LdaModel(pos_corpus,num_topics=3,id2word=pos_dict)#LDA模型训练


# In[74]:


LDA_outnegf='data/haier_jd_neg_LDA_res.txt'
outf_neg=codecs.open(LDA_outnegf,'wb',encoding='utf-8')
LDA_outposf='data/haier_jd_pos_LDA_res.txt'
outf_pos=codecs.open(LDA_outposf,'wb',encoding='utf-8')

for i in range(3):
    outf_pos.write('正向主题%d:\n'%(i+1))
    tmp=pos_lda.print_topic(i)
    print('正向主题%d:\t'%(i+1),tmp,'\n')
    tp=tmp.split(' + ')
    for j in tp:
        a=j.split('*')
        score=a[0].strip()
        word=a[1].strip('\"')
        outf_pos.write(word+'\t'+score+'\n')
    outf_pos.write('\n')

for i in range(3):
    outf_neg.write('负向主题%d:\n'%(i+1))
    tmp=neg_lda.print_topic(i)
    print('负向主题%d:\t'%(i+1),tmp,'\n')
    tp=tmp.split(' + ')
    for j in tp:
        a=j.split('*')
        score=a[0].strip()
        word=a[1].strip('\"')
        outf_neg.write(word+'\t'+score+'\n')
    outf_neg.write('\n')


# In[75]:


import numpy as np
import pandas as pd
import jieba
import re


# In[79]:


data_neg=pd.read_csv('data/haier_jd_neg_2.txt',header=None)
data_neg['label']=0#负向情感标签设为0
#data_neg.columns['words','label']
data_neg.head()


# In[124]:


data_pos=pd.read_csv('data/haier_jd_pos_2.txt',header=None)
data_pos['label']=1#负向情感标签设为0
#data_neg.columns['words','label']
data_pos.head()


# In[81]:


stop=pd.read_csv('data/stoplist.txt',sep='timp',encoding='utf-8',engine='python')


# In[83]:


stop_list = list(stop.iloc[:,0]) + [",", " "]


# In[106]:


import string
desl = string.punctuation
desl = desl + "、（）,"
desl


# In[107]:


data_combine = pd.concat([data_neg, data_pos], axis=0)


# In[108]:


data_combine['words_punc'] = data_combine[0].apply(lambda s : re.sub(r'[%s]+'%desl,"",s))
data_combine['words_cuts'] = data_combine['words_punc'].apply(lambda s : list(jieba.cut(s)))


# In[109]:


data_combine['words_cuts_stop'] = data_combine['words_cuts'].apply(lambda s : [i for i in s if i not in stop_list])
data_combine.head(2)


# In[110]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data_combine['length'] = data_combine['words_cuts_stop'].apply(lambda s : len(s))


# In[111]:


plt.figure(figsize=(6,4))
sns.distplot(data_combine[data_combine['label']==1]['length'],label='pos')
sns.distplot(data_combine[data_combine['label']==0]['length'],label='neg')
plt.legend(['pos','neg'])


# In[112]:


sns.distplot(data_combine[data_combine['label']==1]['length'],label='pos')


# In[113]:


sns.distplot(data_combine[data_combine['label']==0]['length'],label='neg')


# In[114]:


data_combine.pivot_table(index='label',aggfunc={'length':'count'}).plot(kind='bar')
plt.xticks(rotation=0)


# In[125]:


data_combine.pivot_table(index='label',aggfunc={'length':'median'}).plot(kind='bar')
plt.xticks(rotation=0)


# In[116]:


from wordcloud import WordCloud


# In[117]:


data_combine.head(2)


# In[118]:


data_combine['space_words'] = data_combine['words_cuts_stop'].apply(lambda s : ' '.join(s))


# In[126]:


font = r'data/msyh.ttf'


# In[127]:


cloud = WordCloud(width=1240, font_path=font, height=880).generate((' '.join(data_combine[data_combine['label']==0]['space_words'])))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')


# In[123]:


cloud = WordCloud(width=1240, font_path=font, height=880).generate((' '.join(data_combine[data_combine['label']==1]['space_words'])))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')


# In[129]:


from gensim.models import Word2Vec


# In[130]:


emotion_w2v = Word2Vec(data_combine['words_cuts_stop'],min_count=3,size=60)


# In[131]:


emotion_w2v[u'热水器']


# In[132]:


emotion_w2v.most_similar(u'热水器')


# In[133]:


emotion_w2v.similarity(u'热水器',u'办公室')


# In[134]:


from sklearn.model_selection import train_test_split
data_combine.length.describe()


# In[135]:


data_combine.head(2)


# In[136]:


def return_w2v_value(s):
    try:
        return emotion_w2v[s]
    except:
        return [0]*60
def modify_words_length(s):
    if len(s)<=16:
        return s + ['        ']*(16-len(s))
    else:
        return s[:16]


# In[137]:


data_combine['words_cuts_stop_modify'] = data_combine['words_cuts_stop'].apply(lambda s:modify_words_length(s))
data_combine['words_trains_w2v'] = data_combine['words_cuts_stop_modify'].apply(lambda s: [return_w2v_value(i) for i in s])


# In[138]:


data_combine.head(2)


# In[139]:


x_train, x_test, y_train, y_test = train_test_split(data_combine['words_trains_w2v'],data_combine['label'])


# In[140]:


print(x_train.head(2))
print(x_test.head(2))


# In[141]:


x_train_modify = np.array(list(x_train))
x_test_modify = np.array(list(x_test))
y_train_modify = np.array(pd.get_dummies(y_train))
y_test_modify = np.array(pd.get_dummies(y_test))
x_train_modify.shape, x_test_modify.shape, y_train_modify.shape, y_test_modify.shape


# In[143]:


from keras.models import Sequential # 导入神经网络初始化函数
from keras.layers.core import Dense, Activation # 导入神经网络层函数，激活函数
from keras.layers.recurrent import LSTM
import time
start = time.time()

model = Sequential()
model.add(LSTM(22, input_shape=(16, 60),dropout=0.5,activation='sigmoid')) # 22个神经元的LSTM
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())
print('Fit Model..')

history = model.fit(x_train_modify, y_train_modify, batch_size=64,epochs=10)

end = time.time()
print('过程用时 %4.2f 秒' % (end-start))


# In[144]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
from matplotlib import pyplot

mpl.rcParams['font.sans-serif'] = ['SimHei'] # ['Microsoft YaHei'] 制定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是符号'-‘显示为方块的问题

pyplot.plot(history.history['loss'], label='train')
plt.title(u'训练集loss曲线')
plt.xlabel(u'迭代次数')
plt.ylabel(u'loss')
plt.legend() # 显示图例
plt.show() # 显示作图结果
# pyplot.plot(history.history['acc'],label='axx')


# In[145]:


from sklearn.metrics import accuracy_score
prediction = [np.argmax(i) for i in model.predict(x_test_modify)]
# 计算精确率
accuracy_score(prediction, y_test)
