import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

data = pd.read_csv('spam_ham_email_dataset.csv')


def get_sequences(texts, tokenizer, train=True, max_seq_length=None):
    sequences = tokenizer.texts_to_sequences(texts)

    if train:
        max_seq_length = np.max(list(map(lambda x: len(x), sequences)))

    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_length, padding='post')

    return sequences


def preprocess_inputs(df):
    df = df.copy()

    # Split df into X and y
    y = df['label']
    y = y.replace({'ham': 0, 'spam': 1})
    X = df['text']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

    # Create tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=30000)

    # Fit the tokenizer
    tokenizer.fit_on_texts(X_train)

    # Convert texts to sequences
    X_train = get_sequences(X_train, tokenizer, train=True)
    X_test = get_sequences(X_test, tokenizer, train=False, max_seq_length=X_train.shape[1])

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = preprocess_inputs(data)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

inputs = tf.keras.Input(shape=(5029,))

embedding = tf.keras.layers.Embedding(
    input_dim=5000,
    output_dim=64
)(inputs)

flatten = tf.keras.layers.Flatten()(embedding)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)

print(model.summary())
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)
results = model.evaluate(X_test, y_test, verbose=0)

print("    Test Loss: {:.4f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
print("     Test AUC: {:.4f}".format(results[2]))
