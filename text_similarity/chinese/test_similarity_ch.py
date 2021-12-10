# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from dataGenerator_ch import BertSemanticDataGenerator

save_path = r'similarity_model_ch/'
model = tf.saved_model.load(save_path)
labels = ['不相似','相似']

def check_similarity(sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, max_length=128,shuffle=False, include_targets=False,
    )
    # proba = model.predict(test_data[0])[0]
    proba = model(test_data[0])[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]
    return pred, proba

if __name__ == '__main__':
    sentence1 = "我这周五过生日。"
    sentence2 = "这周五我过生日。"
    pred,proba = check_similarity(sentence1, sentence2)
    print(pred)
    print(proba)