# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from dataGenerator import BertSemanticDataGenerator

save_path = r'similarity_model_en/'
model = tf.saved_model.load(save_path)
# model = load_model(save_path)
labels = ["contradiction", "entailment", "neutral"]

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
    sentence1 = "Two women are observing something together."
    sentence2 = "Two women are standing with their eyes closed."
    pred,proba = check_similarity(sentence1, sentence2)
    print(pred)
    print(proba)
