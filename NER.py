import sys
# sys.path.append('/usr/local/lib/python3.9/site-packages')
import numpy as np
from sklearn.model_selection import ShuffleSplit
from data_utils import ENTITIES, Documents, Dataset, SentenceExtractor, make_predictions
from data_utils import Evaluator
from models import build_lstm_crf_model
from gensim.models import Word2Vec
# import gensim
import os
# import time
# import pandas as pd
import keras
data_dir = 'data/'
whether_visualize = False


workdir = os.listdir(data_dir)
if '.DS_Store' in workdir:
  os.remove('./'+data_dir+'.DS_Store')
# print(os.listdir(data_dir))

ent2idx = dict(zip(ENTITIES, range(1, len(ENTITIES) + 1)))
idx2ent = dict([(v, k) for k, v in ent2idx.items()])

docs = Documents(data_dir=data_dir)
rs = ShuffleSplit(n_splits=1, test_size=40, random_state=2022)
train_doc_ids, test_doc_ids = next(rs.split(docs))  # 获得训练和测试的文件列表名称


train_docs, test_docs = docs[train_doc_ids], docs[test_doc_ids]

num_cates = max(ent2idx.values()) + 1   # 16
sent_len = 64
vocab_size = 3000
emb_size = 100
sent_pad = 10
sent_extrator = SentenceExtractor(window_size=sent_len, pad_size=sent_pad)
train_sents = sent_extrator(train_docs)
test_sents = sent_extrator(test_docs)

train_data = Dataset(train_sents, cate2idx=ent2idx)
train_data.build_vocab_dict(vocab_size=vocab_size)

test_data = Dataset(test_sents, word2idx=train_data.word2idx, cate2idx=ent2idx)
vocab_size = len(train_data.word2idx)

w2v_train_sents = []
for doc in docs:
    w2v_train_sents.append(list(doc.text))
w2v_model = Word2Vec(w2v_train_sents, vector_size=emb_size)

w2v_embeddings = np.zeros((vocab_size, emb_size))
for char, char_idx in train_data.word2idx.items():
    if char in w2v_model.wv:
        w2v_embeddings[char_idx] = w2v_model.wv[char]

seq_len = sent_len + 2 * sent_pad
model = build_lstm_crf_model(num_cates, seq_len=seq_len, vocab_size=vocab_size,
                             model_opts={'emb_matrix': w2v_embeddings, 'emb_size': 100, 'emb_trainable': False})
model.summary()

train_X, train_y = train_data[:]
print('train_X.shape', train_X.shape)
print('train_y.shape', train_y.shape)

"""### 训练模型"""

model.fit(train_X, train_y, batch_size=5, epochs=1)

"""### 测试结果"""

preds_train = model.predict(train_X, batch_size=5, verbose=True)
pred_docs_train = make_predictions(preds_train, train_data, sent_pad, docs, idx2ent)
print(pred_docs_train)
f_score, precision, recall = Evaluator.f1_score(train_docs, pred_docs_train)
print('train: f_score: ', f_score)
print('train: precision: ', precision)
print('train: recall: ', recall)

test_X, test_y = test_data[:]
preds_test = model.predict(test_X, batch_size=5, verbose=True)
pred_docs_test = make_predictions(preds_test, test_data, sent_pad, docs, idx2ent)
print(pred_docs_test)
f_score, precision, recall = Evaluator.f1_score(test_docs, pred_docs_test)
print('test: f_score: ', f_score)
print('test: precision: ', precision)
print('test: recall: ', recall)


# # output KB and visualize
# # http://localhost:5000/
# visualize
# http://localhost:5000/
# sample_doc_id = list(pred_docs_test.keys())[0]
# # train_docs[sample_doc_id]._repr_html_()
# pred_docs_test[sample_doc_id]._repr_html_()

import time
ini_port = 5000
file_write_obj = open('output.txt', 'w')
for j in list(pred_docs_test.keys()):
    if whether_visualize:
        while True:
            try:
                time.sleep(2)
                print("{} start".format(j))
                ini_port += 1
                industry_list = pred_docs_test[j]._repr_html_(portid=ini_port, whether_visualize = whether_visualize)
                input("you can visualize on \"http://localhost:{}/\", input anything to continue ...".format(ini_port))
                break
            except (OSError):
                inp = input("retry:0\nnext:1")  # Get the input
                if inp == '1':
                    break
                else:
                    continue
    else:
        while True:
            try:
                print("{} start".format(j))
                ini_port += 1
                industry_list = pred_docs_test[j]._repr_html_(portid=ini_port, whether_visualize = whether_visualize)
                industry_list = list(set(industry_list))
                for i in industry_list:
                    file_write_obj.write(j)
                    file_write_obj.write('\t')
                    file_write_obj.write(i)
                    file_write_obj.write('\n')
                break
            except (OSError):
                inp = input("retry:0\nnext:1")  # Get the input
                if inp == '1':
                    break
                else:
                    continue