"""
Movie Review Sentiment Analysis using LSTM
------------------------------------------
Objective : Movie reviews ko Positive ya Negative classify karna.
Model     : Embedding -> LSTM -> Dropout -> Dense -> Sigmoid
Dataset   : IMDB Movie Reviews (50,000 reviews, Keras mein built-in).
            Pehli baar run karne par automatically download hota hai (internet chahiye).

Install   : pip install tensorflow scikit-learn matplotlib seaborn
Run       : python movie_review_lstm.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------------
VOCAB_SIZE = 10000     # sirf top 10,000 frequent words
MAX_LEN = 200          # har review ki length (tokens)
EMBED_DIM = 128
EPOCHS = 10
BATCH_SIZE = 64
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------------
# 2. Load dataset (25k train + 25k test)
# ------------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
print("Training reviews:", len(X_train))
print("Testing reviews :", len(X_test))

# Ek review decode karke dekhte hain (samajhne ke liye)
word_index = imdb.get_word_index()
# Keras imdb mein indices 3 se shift hote hain (0=pad, 1=start, 2=unknown)
index_to_word = {i + 3: w for w, i in word_index.items()}
index_to_word.update({0: "<PAD>", 1: "<START>", 2: "<UNK>", 3: "<UNUSED>"})

def decode_review(seq):
    return " ".join(index_to_word.get(i, "?") for i in seq)

print("\nSample review:\n", decode_review(X_train[0])[:300], "...")
print("Label:", "Positive" if y_train[0] == 1 else "Negative")

# ------------------------------------------------------------------
# 3. Padding (sab reviews ki length barabar karna)
# ------------------------------------------------------------------
X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding="post", truncating="post")
X_test = pad_sequences(X_test, maxlen=MAX_LEN, padding="post", truncating="post")

# ------------------------------------------------------------------
# 4. Build LSTM model
# ------------------------------------------------------------------
model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),     # 1 = Positive, 0 = Negative
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.build(input_shape=(None, MAX_LEN))
model.summary()

# ------------------------------------------------------------------
# 5. Train
# ------------------------------------------------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1,
)

# ------------------------------------------------------------------
# 6. Evaluate
# ------------------------------------------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss    : {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

y_pred = (model.predict(X_test).ravel() > 0.5).astype(int)
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# Accuracy / Loss graphs
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="val")
ax[0].set_title("Model Accuracy")
ax[0].set_xlabel("Epoch")
ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="val")
ax[1].set_title("Model Loss")
ax[1].set_xlabel("Epoch")
ax[1].legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 7. Predict on custom reviews
# ------------------------------------------------------------------
def encode_review(text):
    """Raw text ko Keras IMDB ke integer format mein convert karta hai."""
    tokens = text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()
    seq = [1]  # <START>
    for w in tokens:
        idx = word_index.get(w)
        seq.append(idx + 3 if idx is not None and idx + 3 < VOCAB_SIZE else 2)  # 2 = <UNK>
    return pad_sequences([seq], maxlen=MAX_LEN, padding="post", truncating="post")

def predict_review(text):
    prob = float(model.predict(encode_review(text), verbose=0)[0][0])
    label = "Positive Review" if prob > 0.5 else "Negative Review"
    return f"{label}  (confidence = {max(prob, 1 - prob):.2%})"

samples = [
    "This movie was fantastic! Great acting and a wonderful story.",
    "Terrible film. Boring plot and awful acting, a complete waste of time.",
    "The visuals were stunning and I loved every minute of it.",
    "I hated this movie, the worst I have ever seen.",
]
print("\n--- Custom predictions ---")
for s in samples:
    print(f"{s}\n  -> {predict_review(s)}\n")

model.save("movie_review_lstm.keras")
print("Model saved as movie_review_lstm.keras")
