import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, log_loss


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

history_dict = pickle.load(open('./history.p', 'rb'))
history = history_dict['history']


data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

y_predict = model.predict(X_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))


# train_pred = model.predict(X_train)
# test_pred = model.predict(X_test)

# # Get probabilities for calculating log loss
# train_probs = model.predict_proba(X_train)
# test_probs = model.predict_proba(X_test)

# # Calculate accuracy for train and test sets
# train_accuracy = accuracy_score(y_train, train_pred)
# test_accuracy = accuracy_score(y_test, test_pred)

# # Calculate log loss for train and test sets
# train_loss = log_loss(y_train, train_probs, labels=np.unique(labels))
# test_loss = log_loss(y_test, test_probs, labels=np.unique(labels))

# # Plotting accuracy and loss
# epochs = range(1)

# plt.figure(figsize=(12, 4))

# # Accuracy plot
# plt.subplot(1, 2, 1)
# plt.plot(epochs, [train_accuracy], 'r', label='Train Accuracy')
# plt.plot(epochs, [test_accuracy], 'b', label='Test Accuracy')
# plt.title('Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# # Loss plot
# plt.subplot(1, 2, 2)
# plt.plot(epochs, [train_loss], 'r', label='Train Loss')
# plt.plot(epochs, [test_loss], 'b', label='Test Loss')
# plt.title('Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# # plt.imsave("Plot.png")
# plt.show()