# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from dataGenerator_ch import BertSemanticDataGenerator

# configuration
max_length = 128  # Maximum length of input sentence to the model.
batch_size = 32
epochs = 2

# There are more than 550k samples in total; we will use 100k for this example.
train_df = pd.read_csv("lcqmc-NLP/lcqmc_train.tsv",delimiter='\t',header=0,dtype={"label": int},nrows=100000)
valid_df = pd.read_csv("lcqmc-NLP/lcqmc_dev.tsv",delimiter='\t',header=0,dtype={"label": int})
test_df = pd.read_csv("lcqmc-NLP/lcqmc_test.tsv",delimiter='\t',header=0,dtype={"label": int})

# Shape of the data
print(f"Total train samples : {train_df.shape[0]}")
print(f"Total validation samples: {valid_df.shape[0]}")
print(f"Total test samples: {valid_df.shape[0]}")

# We have some NaN entries in our train data, we will simply drop them.
print("Number of missing values")
print(train_df.isnull().sum())
train_df.dropna(axis=0, inplace=True)
# Distribution of our training targets.
print("Train Target Distribution")
print(train_df.label.value_counts())
# Distribution of our validation targets.
print("Validation Target Distribution")
print(valid_df.label.value_counts())

# One-hot encode training, validation, and test labels.
y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=2)
y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=2)
y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=2)

# Create the model under a distribution strategy scope.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )
    # Loading pretrained BERT model.
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = False

    bert_output = bert_model(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )
    sequence_output = bert_output.last_hidden_state
    pooled_output = bert_output.pooler_output
    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(sequence_output)
    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(2, activation="softmax")(dropout)
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )


print(f"Strategy: {strategy}")
model.summary()

train_data = BertSemanticDataGenerator(
    train_df[["text_a", "text_b"]].values.astype("str"),
    y_train,
    batch_size=batch_size,
    max_length=max_length,
    shuffle=True,
)
valid_data = BertSemanticDataGenerator(
    valid_df[["text_a", "text_b"]].values.astype("str"),
    y_val,
    batch_size=batch_size,
    max_length=max_length,
    shuffle=False,
)
# train the model
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)
# fine tuning
# Unfreeze the bert_model.
bert_model.trainable = True
# Recompile the model to make the change effective.
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# Train the entire model end-to-end
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)

model_path = "/home/iie/LRQ/similarity_model_ch/"
tf.saved_model.save(model, model_path)
