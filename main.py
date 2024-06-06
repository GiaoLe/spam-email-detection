import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import logging
import string
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Set up logging
logging.basicConfig(level=logging.INFO)

# Here we download the stopwords package
nltk.download('stopwords')
from nltk.corpus import stopwords

data = pd.read_csv('Spam Email raw text for NLP.csv')

stop_words = set(stopwords.words('english'))

# Number of words to use in the model.
# We will use the 30000 most frequent words
# The number is often chose between 10000 and 30000
# because having a lots of words can make the model confused
num_words = 30000


def remove_stop_words(sentence):
    words = sentence.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


# Function to convert texts to sequences to use them in the model
def get_sequences(texts, tokenizer, train=True, max_seq_length=None):
    sequences = tokenizer.texts_to_sequences(texts)

    # If training, we want to know the maximum sequence length and pad 0 accordingly
    if train:
        max_seq_length = np.max(list(map(lambda x: len(x), sequences)))
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_length, padding='post')

    return sequences


df = data.copy()

# Split df into X and y
y = df['CATEGORY']
X = df['MESSAGE']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

# Create tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)

# Fit the tokenizer
tokenizer.fit_on_texts(X_train)

logging.info('Number of words: %s', len(tokenizer.word_index))

# Preprocess the data
# Stop words removal
X = pd.Series(X)
X = X.apply(remove_stop_words)


# Text normalization
def normalize_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text


X = X.apply(normalize_text)

# Convert texts to sequences
X_train = get_sequences(X_train, tokenizer, train=True)
X_test = get_sequences(X_test, tokenizer, train=False, max_seq_length=X_train.shape[1])

logging.info('X_train shape: %s', X_train.shape)

# # Create the model
inputs = tf.keras.Input(shape=(X_train.shape[1],))

# We choose the output here to be 64. Need more analysis to choose the best value
embedding = tf.keras.layers.Embedding(input_dim=num_words,
                                      output_dim=64
                                      )(inputs)

flatten = tf.keras.layers.Flatten()(embedding)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(
    tf.keras.layers.Dense(64, activation='relu')(tf.keras.layers.Dense(32, activation='relu')(flatten)))

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
# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

# Plot the accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Predict probabilities for the test data.
y_pred_probs = model.predict(X_test)

# Compute the ROC curve.
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Compute the AUC (Area Under the Curve).
roc_auc = auc(fpr, tpr)

# Plot the ROC curve.
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Evaluate the model on the test data
results = model.evaluate(X_test, y_test, verbose=0)

print("    Test Loss: {:.4f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
print("     Test AUC: {:.4f}".format(results[2]))

# Predict probabilities for the test data.
y_pred_probs = model.predict(X_test)

# Convert probabilities to binary predictions
y_pred = np.squeeze(y_pred_probs) >= 0.5

# Compare predictions with actual results
incorrect_predictions = y_pred != y_test

# Count the number of incorrect predictions
num_incorrect_predictions = incorrect_predictions.sum()

print(f"Total number of incorrect predictions: {num_incorrect_predictions}")

model.save('spam_email_classifier.keras')
