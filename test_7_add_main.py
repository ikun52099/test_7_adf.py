import os
import jieba
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import LSTMCell, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dropout, RNN, Bidirectional, LSTM
from keras.callbacks import ModelCheckpoint

print(1)
# 初始化tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese/')
print(2)

def load_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]
    return texts

# 加载数据
AD_texts = load_texts('data/AD.txt')
HC_texts = load_texts('data/HC.txt')
MCI_texts = load_texts('data/MCI.txt')

# 分词
AD_texts_tokenized = [' '.join(jieba.cut(text)) for text in AD_texts]
HC_texts_tokenized = [' '.join(jieba.cut(text)) for text in HC_texts]
MCI_texts_tokenized = [' '.join(jieba.cut(text)) for text in MCI_texts]

# 构建数据列表
texts = AD_texts_tokenized + HC_texts_tokenized + MCI_texts_tokenized
labels = [0] * len(AD_texts) + [1] * len(HC_texts) + [2] * len(MCI_texts)

# 对数据进行切分
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)



train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
print(type(train_encodings))

# 创建tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))


train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# 创建tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))

if 1:
    # 定义BERT模型并抽取特征
    bert_model = TFBertModel.from_pretrained('bert-base-chinese')
    
    input_ids = Input(shape=(None,), dtype=tf.int32, name='input_ids')  # 假设最大长度为512
    attention_mask = Input(shape=(None,), dtype=tf.int32, name='attention_mask')

    sequence_output = bert_model([input_ids, attention_mask])[0]
    
    # 添加双向LSTM层
    lstm_layer = Bidirectional(LSTM(units=32, return_sequences=True))  # LSTM单元数可自定义
    lstm_out = lstm_layer(sequence_output)

    # 添加全连接层（线性层）
    dense_layer = Dense(units=16, activation='relu')###################
    dense_out = dense_layer(lstm_out)

    dense_out = tf.reduce_mean(dense_out, axis=1, keepdims=False)

    # 添加输出层与softmax激活函数
    output_layer = Dense(units=3)  # 输出类别数为3
    output = output_layer(dense_out)

    # 构建完整模型
    model = Model(inputs=[input_ids, attention_mask], outputs=output)

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    # 训练模型
    history = model.fit(train_dataset.shuffle(100).batch(16), epochs=15, validation_data=val_dataset.batch(1))

    # 保存模型
    model.save('my_model_add.h5')

def load():
    global model
    # 加载模型并进行推理
    model = load_model('my_model_add.h5')

def pre(t):
    # 假设有个新的文本需要预测
    new_text = [t]
    new_text_tokenized = [' '.join(jieba.cut(text)) for text in new_text]
    new_encoding = tokenizer(new_text_tokenized, truncation=True, padding=True, max_length=512, return_tensors='tf')

    prediction = model.predict(x=[new_encoding['input_ids'], new_encoding['attention_mask']], verbose=0)
    return int(tf.argmax(prediction[0], axis=-1).numpy())
