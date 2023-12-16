# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt


def readMNISTdata():
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data / 256, test_labels

def softmax(scores):
    log_denominator = np.log(np.sum(np.exp(scores - np.max(scores, axis=1, keepdims=True)), axis=1, keepdims=True))
    log_probs = scores - np.max(scores, axis=1, keepdims=True) - log_denominator
    return np.exp(log_probs)

def encode_categorical(x, num_classes):
    m = x.shape[0]
    encoded = np.zeros((m, num_classes))
    encoded[np.arange(m), x[:, 0]] = 1
    return encoded

def compute_gradient(X, y, t, W):
    N = X.shape[0]
    grad = np.dot(X.T, (y - encode_categorical(t, W.shape[1]))) / N
    return grad

def predict(X, W, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here
    scores = np.dot(X, W)
    y = softmax(scores)

    t_hat = np.argmax(y, axis=1).reshape(-1, 1)

    loss = None
    acc = None
    if t is not None:
        correct_log_probs = -np.log(y[range(len(t)), t.squeeze()])
        loss = np.sum(correct_log_probs) / len(t)
        acc = np.mean(t_hat == t)

    return y, t_hat, loss, acc


def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]
    K = np.unique(t_train).size  
    W = np.random.randn(X_train.shape[1], K)

    # TODO Your code here
    train_losses = []
    valid_accs = []
    acc_best = 0
    epoch_best = 0
    W_best = None

    for epoch in range(MaxEpoch):
        
        permutation = np.random.permutation(N_train)
        X_train_shuffled = X_train[permutation]
        t_train_shuffled = t_train[permutation]

        for i in range(0, N_train, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            t_batch = t_train_shuffled[i:i+batch_size]

    
            y_batch, _, _, _ = predict(X_batch, W, t_batch)
            grad = compute_gradient(X_batch, y_batch, t_batch, W)
            W -= alpha * grad

        _, _, train_loss, _ = predict(X_train, W, t_train)
        train_losses.append(train_loss)

        _, _, _, acc_value = predict(X_val, W, t_val)
        valid_accs.append(acc_value)

        if acc_value > acc_best:
            acc_best = acc_value
            epoch_best = epoch
            W_best = W.copy()

        print(f'Epoch {epoch + 1}/{MaxEpoch} - Loss: {train_loss:.4f}, Validation Accuracy: {acc_value:.4f}')

    return epoch_best, acc_best,  W_best, train_losses, valid_accs


##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


print(X_train.shape, t_train.shape, X_val.shape,
      t_val.shape, X_test.shape, t_test.shape)


N_class = 10

alpha = 1    # learning rate
batch_size = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay


# TODO: report 3 number, plot 2 curves
epoch_best, acc_best,  W_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val)

_, _, _, acc_test = predict(X_test, W_best, t_test)

print(f'Best Epoch: {epoch_best}')
print(f'Best Validation Accuracy: {acc_best}')
print(f'Test Accuracy: {acc_test}')


# Plot training loss

plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('q2 experiment training loss' + '.jpg')

plt.figure()

# Plot validation accuracy
plt.plot(valid_accs, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy ')
plt.legend()
plt.savefig('q2 experiment valid acc' + '.jpg')

