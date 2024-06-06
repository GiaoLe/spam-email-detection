import numpy as np
import pandas as pd
import tensorflow as tf
from main import remove_stop_words, get_sequences, tokenizer, X_train, normalize_text

model = tf.keras.models.load_model('spam_email_classifier.keras')


def predict_new_email(model, tokenizer, email):
    # Preprocess the email
    email = remove_stop_words(email)
    email = normalize_text(email)

    # Convert the email to sequences
    sequences = get_sequences(pd.Series([email]), tokenizer, train=False, max_seq_length=X_train.shape[1])

    # Predict whether the email is spam or not
    pred_probs = model.predict(sequences)

    # Convert probabilities to binary predictions
    pred = np.squeeze(pred_probs) >= 0.5

    return 'spam' if pred else 'ham'


# Load the dataset
data = pd.read_csv('spam_ham_email_dataset.csv')

# Randomly sample 5 emails
samples = data.sample(5)

# Get the actual labels
actual_labels = samples['label']
actual_labels.map({'ham': 0, 'spam': 1})
# Get the email texts
emails = samples['text']

# Initialize a counter for correct predictions
correct_predictions = 0

# For each sampled email
for email, actual_label in zip(emails, actual_labels):
    # Predict whether the email is spam or not
    predicted_label = predict_new_email(model, tokenizer, email)

    # Check if the prediction is correct
    is_correct = (predicted_label == actual_label)

    # If the prediction is correct, increment the counter
    if is_correct:
        correct_predictions += 1

    # Print the model's prediction, the actual label, and whether the prediction was correct
    print(f"Email: {email}")
    print(f"Predicted label: {predicted_label}")
    print(f"Actual label: {actual_label}")
    print(f"Correct prediction: {is_correct}")
    print()

# Print the total number of correct predictions
print(f"Total number of correct predictions: {correct_predictions}")