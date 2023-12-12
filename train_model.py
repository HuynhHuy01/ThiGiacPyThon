import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense
import seaborn as sns
DATA_PATH = os.path.join('MP_Data')
actions = np.array(["father","hello","I","mother","see_you_later","what","again"])

#30 video chứa data
no_sequences = 30

#video sẽ mang 30 frames
sequence_length = 30
actions = np.array(["father","hello","I","mother","see_you_later","what","again"])
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(np.array(sequences).shape)
print(y_test.shape)


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tb_callback = TensorBoard(log_dir=os.path.join('Logs'))
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0001)
model = Sequential()
model.add(LSTM(16, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(32, return_sequences=False, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs= 150, callbacks=[tb_callback])
# validation_data=(X_val,y_val)
model.save("newlytrained.keras")
plt.plot(history.history['categorical_accuracy'], label='Train Categorical Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Categorical Accuracy')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='Train Loss', color = "red")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
print(multilabel_confusion_matrix(ytrue, yhat))
accuracy = accuracy_score(ytrue, yhat)
precision = precision_score(ytrue, yhat, average='weighted')
recall = recall_score(ytrue, yhat, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

conf_mat = multilabel_confusion_matrix(ytrue, yhat)
class_names = ["father","hello","I","mother","see_you_later","what","again"]
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 10))  # Adjusted dimensions
axes = axes.flatten()  # Flatten the array for easier indexing
for i, class_name in enumerate(class_names):
    sns.heatmap(conf_mat[i], annot=True, fmt='d', cmap='viridis',
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                annot_kws={"size": 16},
                ax=axes[i])  # Use i instead of [row, col]
    axes[i].set_title(f'Confusion Matrix for Class {class_name}')
    axes[i].set_xlabel("Predicted Label")
    axes[i].set_ylabel("True Label")
plt.tight_layout()
plt.show()
# res = model.predict(X_test)
# print(actions[np.argmax(res[0])])
# print(actions[np.argmax(y_test[0])])
# y_pred = model.predict(X_train)
# y_true = np.argmax(y_train, axis=1).tolist()
# y_pred = np.argmax(y_pred, axis=1).tolist()
