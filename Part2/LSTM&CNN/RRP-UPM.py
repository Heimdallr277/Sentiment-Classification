
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, LSTM
from tensorflow.keras import Model

doc_len = 5
sen_len = 10
hidden_size = 100


# In[3]:


# 读取词嵌入文件
def read_embedfile(embedding_file):
    # 建立词与词向量的映射表
    words_dict = dict()
    words_dict['$EOF$'] = np.zeros(100)
    lines = []
    with open(embedding_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        f.close()
    lines = lines[1:]
    for line in lines:
        line = line.strip().split()
        vec = [float(s) for s in line[1:]]
        words_dict[line[0]] = np.array(vec)

    return words_dict


# In[4]:


# 首先是数据处理
def read_file(filepath, words_dict):
    dataset = pd.read_csv(filepath, sep='\t\t', engine='python', encoding='utf-8')
    dataset.columns = ['user', 'product', 'rating', 'doc']

    # 将用户转换为向量
    user_dict = {}
    user_vec = []
    for i in range(len(dataset['user'])):
        word = dataset['user'][i]
        if word not in user_dict:
            user_dict[word] = np.random.normal(0, 1, 100) # 用户向量维度为50
        user_vec.append(user_dict[word])
    dataset['user'] = user_vec

    # 将物品转换为向量
    product_dict = {}
    product_vec = []
    for i in range(len(dataset['product'])):
        word = dataset['product'][i]
        if word not in product_dict:
            product_dict[word] = np.random.normal(0, 1, 100)  # 用户向量维度为100
        product_vec.append(product_dict[word])
    dataset['product'] = product_vec

    # 将文本转换成词向量组成的矩阵
    # 写个大概，所以只是简单设置文长和句长
    doc_vec = []
    for i in range(len(dataset['doc'])):
        doc = np.zeros((doc_len, sen_len, 100))
        text = dataset['doc'][i]
        sentences = text.strip().split('<sssss>')
        sentences_len = []
        for j in range(len(sentences)):
            sentences[j] = sentences[j].split()
            sentences_len.append(len(sentences[j]))

        # 选择词数最多的5个句子
        new_sentences = []
        index = np.argsort(sentences_len)
        index = index[::-1]
        for j in range(doc_len):
            if j >= len(sentences):
                new_sentences.append(['$EOF$'] * sen_len)
                continue
            if sentences_len[index[j]] < sen_len:
                sentences[index[j]] += ['$EOF$'] * sen_len
            new_sentences.append(sentences[index[j]][:sen_len])

        for j in range(doc_len):
            for x in range(sen_len):

                if new_sentences[j][x] not in words_dict:
                    new_sentences[j][x] = '$EOF$'
                doc[j][x] = words_dict[new_sentences[j][x]]
        doc_vec.append(doc)
    dataset['doc'] = doc_vec
    return dataset


# In[5]:


# 数据载入
word_dict = read_embedfile('data/embedding.txt')
trainset = read_file('data/train.ss', word_dict)
testset = read_file('data/test.ss', word_dict)


# In[6]:


trainset.head(5)


# In[14]:


train_inputs = np.concatenate(trainset['doc'].values).reshape((train_len,doc_len,sen_len,100))
train_users = np.concatenate(trainset['user'].values).reshape((train_len,100))
train_products = np.concatenate(trainset['product'].values).reshape((train_len,100))
train_x = [train_inputs, train_users, train_products]
train_y = trainset['rating'].values
for i in range(len(train_y)):
    train_y[i] = train_y[i]-1
test_inputs = np.concatenate(testset['doc'].values).reshape((test_len,doc_len,sen_len,100))
test_users = np.concatenate(testset['user'].values).reshape((test_len,100))
test_products = np.concatenate(testset['product'].values).reshape((test_len,100))
test_x = [test_inputs, test_users, test_products]
test_y = testset['rating'].values
for i in range(len(test_y)):
    test_y[i] = test_y[i]-1


# In[15]:


# 模型
class RRP_UPM(Model):
    def __init__(self):
        super(RRP_UPM, self).__init__()
        self.cnn_s = Conv2D(filters=10, kernel_size=(3, 3), padding='same')
        self.lstm_s = LSTM(hidden_size)
        self.cnn_d = Conv2D(filters=20, kernel_size=(3, 3), padding='same')
        self.lstm_d = LSTM(hidden_size)
        self.dense = Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
        
        #创建一个可训练的权重变量矩阵
        self.u_wh_1 = self.add_weight(name='u_wh_1',shape=((hidden_size,hidden_size)),initializer='uniform',trainable=True)
        self.u_wh_2 = self.add_weight(name='u_wh_2',shape=((hidden_size,hidden_size)),initializer='uniform',trainable=True)
        self.u_wh_1_b = self.add_weight(name='u_wh_1_b',shape=((1,hidden_size)),initializer='uniform',trainable=True)
        self.u_wh_2_b = self.add_weight(name='u_wh_2_b',shape=((1,hidden_size)),initializer='uniform',trainable=True)
        self.u_v_1 = self.add_weight(name='u_v_1',shape=((hidden_size,hidden_size)),initializer='uniform',trainable=True)
        self.u_v_2 = self.add_weight(name='u_v_2',shape=((hidden_size,hidden_size)),initializer='uniform',trainable=True)
        self.p_wh_1 = self.add_weight(name='p_wh_1',shape=((hidden_size,hidden_size)),initializer='uniform',trainable=True)
        self.p_wh_2 = self.add_weight(name='p_wh_2',shape=((hidden_size,hidden_size)),initializer='uniform',trainable=True)    
        self.p_wh_1_b = self.add_weight(name='p_wh_1_b',shape=((1,hidden_size)),initializer='uniform',trainable=True)
        self.p_wh_2_b = self.add_weight(name='p_wh_2_b',shape=((1,hidden_size)),initializer='uniform',trainable=True)
        self.p_v_1 = self.add_weight(name='p_v_1',shape=((hidden_size,hidden_size)),initializer='uniform',trainable=True)
        self.p_v_2 = self.add_weight(name='p_v_2',shape=((hidden_size,hidden_size)),initializer='uniform',trainable=True)
        self.wu_1 = self.add_weight(name='wu_1',shape=((hidden_size,hidden_size)),initializer='uniform',trainable=True)
        self.wu_2 = self.add_weight(name='wu_2',shape=((hidden_size,hidden_size)),initializer='uniform',trainable=True)
        self.wp_1 = self.add_weight(name='wp_1',shape=((hidden_size,hidden_size)),initializer='uniform',trainable=True)
        self.wp_2 = self.add_weight(name='wp_2',shape=((hidden_size,hidden_size)),initializer='uniform',trainable=True)
        
    def call(self, dataset):
        
        x, user, product = dataset
        
        # word level
        y1_u = self.cnn_s(x) 
        y2_u = self.lstm_s(tf.reshape(x,[-1,sen_len,100])) 
        y1_u = tf.reshape(y1_u,[-1,100])
        y2_u = tf.reshape(y2_u,[-1,100])
        y_u = y1_u+y2_u
        
        u = tf.matmul(y_u,self.u_wh_1)+self.u_wh_1_b
        u = tf.reshape(u,[-1,doc_len,100])
        u += tf.cast(tf.matmul(user, self.wu_1)[:,None,:],dtype=tf.float32)
        u = tf.matmul(tf.tanh(u),self.u_v_1)
        u = tf.reshape(u,[-1,doc_len,100])
        alpha = tf.nn.softmax(u)
        alpha = tf.reshape(alpha,[-1,100])
        s_r_u = alpha * y_u
        sen_re_u = tf.reshape(s_r_u,[-1,doc_len,100]) 
        
        y1_p = self.cnn_s(x) 
        y2_p = self.lstm_s(tf.reshape(x,(-1,sen_len,100))) 
        y1_p = tf.reshape(y1_p,[-1,100])
        y2_p = tf.reshape(y2_p,[-1,100])
        y_p = y1_p+y2_p
        
        p = tf.matmul(y_p,self.p_wh_1)+self.p_wh_1_b
        p = tf.reshape(p,[-1,doc_len,100])
        p += tf.cast(tf.matmul(product, self.wp_1)[:,None,:],dtype=tf.float32)
        p = tf.matmul(tf.tanh(p),self.p_v_1)
        p = tf.reshape(p,(-1,doc_len,100))
        alpha = tf.nn.softmax(p)
        alpha = tf.reshape(alpha,(-1,100))
        s_r_p = alpha * y_p
        sen_re_p = tf.reshape(s_r_p,(-1,doc_len,100))
        
        # sentence level
        y1_u = self.cnn_d(tf.reshape(sen_re_u,((-1,1,doc_len,100)))) # 要四个参数
        y2_u = self.lstm_d(tf.reshape(sen_re_u,((-1,doc_len,100)))) # 要三个参数
        y1_u = tf.reshape(y1_u,(-1,100))
        y2_u = tf.reshape(y2_u,(-1,100))
        y_u = y1_u+y2_u
        
        u = tf.matmul(y_u,self.u_wh_2)+self.u_wh_2_b
        u += tf.cast(tf.matmul(user, self.wu_2),dtype=tf.float32)
        u = tf.matmul(tf.tanh(u),self.u_v_2)
        alpha = tf.nn.softmax(u)
        d_r_u = alpha * y_u
        doc_re_u = tf.reshape(d_r_u,(-1,100))
        
        y1_p = self.cnn_d(tf.reshape(sen_re_p,((-1,1,doc_len,100)))) # 要四个参数
        y2_p = self.lstm_d(tf.reshape(sen_re_p,((-1,doc_len,100)))) # 要三个参数
        y1_p = tf.reshape(y1_p,(-1,100))
        y2_p = tf.reshape(y2_p,(-1,100))
        y_p = y1_p+y2_p
        
        p = tf.matmul(y_p,self.p_wh_2)+self.p_wh_2_b
        p += tf.cast(tf.matmul(product, self.wp_2),dtype=tf.float32)
        p = tf.matmul(tf.tanh(p),self.p_v_2)
        alpha = tf.nn.softmax(p)
        d_r_p = alpha * y_p
        doc_re_p = tf.reshape(d_r_p,[-1,100]) 
        
        doc_vec = tf.concat([doc_re_u,doc_re_p],1)
        result = self.dense(doc_vec)
        
        return result


# In[16]:


model = RRP_UPM()

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
             metrics=['sparse_categorical_accuracy'])

#model.fit(train_x, train_y, batch_size=32, epochs=10, validation_data=(test_x, test_y))
model.fit(train_x, train_y, batch_size=32, epochs=10, validation_split=0.2)
model.summary()

