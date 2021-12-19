import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import jieba as jb
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
matplotlib.style.use("classic")
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report


final_stock_list = ['600000', '600009', '600016', '600031', '600036', '600048', '600309', '600547', '600570', '600585', 
                    '600588', '600690', '600703', '600745', '600809', '600887', '600893', '601088', '601138', '601288', 
                    '601336', '601398', '601601', '601688', '601818', '601857', '601888', '603259', '603288', '603501', '603986']
final_stock_name = ["浦发银行", "上海机场", "民生银行", "三一重工", "招商银行", "保利地产", "万华化学", "山东黄金", "恒生电子", "海螺水泥",
                    "用友网络", "海尔智家", "三安光电", "闻泰科技", "山西汾酒", "伊利股份", "航发动力", "中国神华", "工业富联", "农业银行", 
                    "新华保险", "工商银行", "中国太保", "华泰证券", "光大银行", "中国石油", "中国国旅", "药明康德", "海天味业", "韦尔股份", "兆易创新"]

total_data_tv = pd.DataFrame(columns = ["Content", "Tag"])

for i in tqdm(range(len(final_stock_list))):
    train_data = pd.read_csv(f"data_cleaned/Train/{final_stock_list[i]}NLP_Train_cleaned.csv")
    valid_data = pd.read_csv(f"data_cleaned/Valid/{final_stock_list[i]}NLP_Valid_cleaned.csv")
    for k in range(len(train_data)):
        content = train_data["Content"].loc[k]
        tag = train_data["Tag"].loc[k]
        if tag == -2:
            train_data["Tag"].loc[k] = "0"
        elif tag == -1:
            train_data["Tag"].loc[k] = "1"
        elif tag == 0:
            train_data["Tag"].loc[k] = "2"
        elif tag == 1:
            train_data["Tag"].loc[k] = "3"
        else:
            train_data["Tag"].loc[k] = "4"
        content = content.replace(final_stock_name[i], "")
        train_data["Content"].loc[k] = content
    for j in range(len(valid_data)):
        content = valid_data["Content"].loc[j]
        tag = valid_data["Tag"].loc[j]
        if tag == -2:
            valid_data["Tag"].loc[j] = "0"
        elif tag == -1:
            valid_data["Tag"].loc[j] = "1"
        elif tag == 0:
            valid_data["Tag"].loc[j] = "2"
        elif tag == 1:
            valid_data["Tag"].loc[j] = "3"
        else:
            valid_data["Tag"].loc[j] = "4"
        content = content.replace(final_stock_name[i], "")
        valid_data["Content"].loc[j] = content
    total_data_tv = total_data_tv.append(train_data[["Content", "Tag"]])
    total_data_tv = total_data_tv.append(valid_data[["Content", "Tag"]])
    
total_data_tv = total_data_tv.reset_index(drop=True)


d = {'Tag':total_data_tv['Tag'].value_counts().index, 'count': total_data_tv['Tag'].value_counts()}
df_cat = pd.DataFrame(data=d).reset_index(drop=True)


df_cat.plot(x='Tag', y='count', kind='bar', legend=False,  figsize=(8, 5))
plt.title("Category Distribution")
plt.ylabel('Number', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()

stopword_path = 'data_cleaned/chinese_stopwords.txt'
# create stop word list, punctuations are included in it
def get_stop_words():
    stopwords = [line.strip() for line in open(stopword_path, encoding='UTF-8').readlines()]
    stopwords += [",", "'"]
    return stopwords
stopwords = get_stop_words()


total_data_tv['Content'] = total_data_tv['Content'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
print(total_data_tv.head())


# set the most frequently used 1000 words
MAX_NB_WORDS = 1000
# cut_review max length
MAX_SEQUENCE_LENGTH = 250
# Embedding dimension
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(total_data_tv['Content'].values)
word_index = tokenizer.word_index
print('There are %s different words.' % len(word_index))


X = tokenizer.texts_to_sequences(total_data_tv['Content'].values)
# fill in the blank of X, and make the length of X equal
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# turn multiclass label into onehot
Y = pd.get_dummies(total_data_tv['Tag']).values
print(X.shape)
print(Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 615)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# Build the model
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


epochs = 8
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis = 1)
Y_test = Y_test.argmax(axis = 1)


conf_mat = confusion_matrix(Y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=["0","1","2","3","4"], yticklabels=["0","1","2","3","4"])
plt.ylabel('Ground Truth',fontsize=18)
plt.xlabel('Prediction',fontsize=18)
plt.show()

 
print('accuracy %s' % accuracy_score(y_pred, Y_test))
print(classification_report(Y_test, y_pred,target_names=["0","1","2","3","4"]))


def predict(text):
    txt = text
    txt = [" ".join([w for w in list(jb.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    cat_id= pred.argmax(axis=1)[0]
    return str(cat_id)

# The Bactesting

pd1 = pd.read_excel("Stock_Data/600000.xlsx").T
pd1 = pd1.set_axis(pd1.iloc[0],axis=1,inplace= False)
pd1 = pd1.drop(index="Unnamed: 0")
date = list(pd1.index.values)
date = [ x.strftime("%Y-%m-%d") for x in date ]
data_context = pd.DataFrame(index=date, columns=final_stock_list)

for k in range(len(final_stock_list)):
    data = pd.read_csv(f"data_cleaned/Test/{final_stock_list[k]}NLP_Test_cleaned.csv")
    for i in range(len(data)):
        if data["Time"].iloc[i] in date:
            data_context[final_stock_list[k]].loc[data["Time"].iloc[i]] = data["Content"].iloc[i].replace(final_stock_name[k], "")

data_context.fillna(value="0",inplace=True)
data_context = data_context.applymap(lambda x: predict(x))
data_context = data_context.applymap(lambda x: int(x)-2)

# Now, we have already get the rank data
data_deca = data_context.copy()
data_deca = data_deca.rolling(7, min_periods=1).mean()
data_deca = data_deca.add(data_context, fill_value=0)
data_deca['max_idx'] = data_deca.idxmax(axis=1)

for i in final_stock_list:
    data = pd.read_excel(f"Stock_data/{i}.xlsx").T
    data = data.set_axis(data.iloc[0],axis=1,inplace= False)  
    data = data.drop(index="Unnamed: 0")
    data["return"] = (data["close"].shift(-1) - data["close"]) / data["close"]
    data.dropna(inplace=True)
    index = [x.strftime("%Y-%m-%d") for x in data.index.values]
    data["Time"] = index
    data = data.set_index("Time", drop=True)
    data.to_csv(f"Stock_data_c/{i}.csv", index=True)

return_r_l = []
for i in data_deca.index.values[:-1]:
    stock = data_deca['max_idx'].loc[i]
    data = pd.read_csv(f"Stock_data_c/{stock}.csv", index_col=0)
    return_r = data["return"].loc[i] + 1
    return_r_l.append(return_r)

ss_r_l = [return_r_l[0]]
for i in range(1, len(return_r_l)):
    ss_r = return_r_l[i] + ss_r_l[i-1]
    ss_r_l.append(ss_r)

j = []
a = 1
for i in return_r_l:
    a *= i
    j.append(a)

b_r_l = []
for i in data_deca.index.values[:-1]:
    b_r = 0
    for stock in final_stock_list:
        data = pd.read_csv(f"Stock_data_c/{stock}.csv", index_col=0)
        b_r += data["return"].loc[i]
    b_r = b_r/len(final_stock_list) + 1
    b_r_l.append(b_r)

l = []
b = 1
for i in b_r_l:
    b *= i
    l.append(b)


data_deca_r3 = data_context.copy()
data_deca_r3 = data_deca_r3.rolling(7, min_periods=1).mean()
data_deca_r3 = data_deca_r3.add(data_context, fill_value=0)
data_deca_r3_t = data_deca_r3.T
rslt = pd.DataFrame(np.zeros((0,3)), columns=['top1','top2','top3'])
for i in data_deca_r3_t.columns:
    df1row = pd.DataFrame(data_deca_r3_t.nlargest(3, i).index.tolist(), index=['top1','top2','top3']).T
    rslt = pd.concat([rslt, df1row], axis=0)
rslt = rslt.set_index(data_deca_r3.index.values, drop=True)

return_r_l3 = []
for i in rslt.index.values[:-1]:
    stock1 = rslt['top1'].loc[i]
    stock2 = rslt['top2'].loc[i]
    stock3 = rslt['top3'].loc[i]
    data1 = pd.read_csv(f"Stock_data_c/{stock1}.csv", index_col=0)
    data2 = pd.read_csv(f"Stock_data_c/{stock2}.csv", index_col=0)
    data3 = pd.read_csv(f"Stock_data_c/{stock3}.csv", index_col=0)
    return_r3_1 = data1["return"].loc[i] + 1
    return_r3_2 = data2["return"].loc[i] + 1
    return_r3_3 = data3["return"].loc[i] + 1
    return_m3 = (return_r3_1 + return_r3_2 + return_r3_3)/3
    return_r_l3.append(return_m3)

h = []
q = 1
for i in return_r_l3:
    q *= i
    h.append(q)

# rank 5
data_deca_r5 = data_context.copy()
data_deca_r5 = data_deca_r5.rolling(7, min_periods=1).mean()
data_deca_r5 = data_deca_r5.add(data_context, fill_value=0)
data_deca_r5_t = data_deca_r5.T
rslt5 = pd.DataFrame(np.zeros((0,5)), columns=['top1','top2','top3','top4','top5'])

for i in data_deca_r5_t.columns:
    df1row = pd.DataFrame(data_deca_r5_t.nlargest(5, i).index.tolist(), index=['top1','top2','top3','top4','top5']).T
    rslt5 = pd.concat([rslt5, df1row], axis=0)
rslt5 = rslt5.set_index(data_deca_r5.index.values, drop=True)

return_r_l5 = []
for i in rslt5.index.values[:-1]:
    stock1 = rslt5['top1'].loc[i]
    stock2 = rslt5['top2'].loc[i]
    stock3 = rslt5['top3'].loc[i]
    stock4 = rslt5['top4'].loc[i]
    stock5 = rslt5['top5'].loc[i]
    data1 = pd.read_csv(f"Stock_data_c/{stock1}.csv", index_col=0)
    data2 = pd.read_csv(f"Stock_data_c/{stock2}.csv", index_col=0)
    data3 = pd.read_csv(f"Stock_data_c/{stock3}.csv", index_col=0)
    data4 = pd.read_csv(f"Stock_data_c/{stock4}.csv", index_col=0)
    data5 = pd.read_csv(f"Stock_data_c/{stock5}.csv", index_col=0)
    return_r5_1 = data1["return"].loc[i] + 1
    return_r5_2 = data2["return"].loc[i] + 1
    return_r5_3 = data3["return"].loc[i] + 1
    return_r5_4 = data4["return"].loc[i] + 1
    return_r5_5 = data5["return"].loc[i] + 1
    return_m5 = (return_r5_1 + return_r5_2 + return_r5_3 +return_r5_1 +return_r5_1)/5
    return_r_l5.append(return_m5)

m = []
t = 1
for i in return_r_l5:
    t *= i
    m.append(t)

plt.figure(figsize=(8,4))
plt.plot(l, label="b")
plt.plot(j, label="r1")
plt.plot(h, label="r3")
plt.plot(m, label="r5")
plt.title("Back Testing Result")
plt.legend(loc='upper left')

print(f"The excess return get by Rank3 factor is {(h[-1] - l[-1]) * 12 / 5}")
print(f"The excess return get by Rank5 factor is {(m[-1] - l[-1]) * 12 / 5}")

data_r = data_context.copy()

for stock in data_context.columns.to_list():
    data_s = pd.read_csv(f"Stock_data_c/{stock}.csv", index_col=0)
    data_r[stock] = data_s["return"]
data_r = data_r.dropna()


data_context_ic3 = data_deca_r3.iloc[:-1,:]

IC_l = []
for stock in data_context_ic3.columns.values:
    a = np.corrcoef(data_r[stock], data_context_ic3[stock])
    IC_l.append(a[0][1])

print(f"The IC of factors is {np.mean(IC_l)}")
print(f"The IR of factors is {np.mean(IC_l)/np.std(IC_l)}")
