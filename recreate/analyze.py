"""Analyze the results of the recreation experiment."""

import json
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def main():

    # Load the results.
    batch_size = 64
    with open('metrics.json', 'r') as f:
        results = json.load(f)

    # Calculate moving averages.
    gab_loss = moving_average(results['gabornet_loss'], 10)
    gab_correct = moving_average(results['gabornet_correct'], 10) / batch_size
    cnn_loss = moving_average(results['cnn_loss'], 10)
    cnn_correct = moving_average(results['cnn_correct'], 10) / batch_size

    # Plot the losses.
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))

    ax[0].plot(results['gabornet_loss'], c='orange', alpha=0.2)
    ax[0].plot(results['cnn_loss'], c='blue', alpha=0.2)

    ax[0].plot(gab_loss, label='gabornet', c='orange')
    ax[0].plot(cnn_loss, label='cnn', c='blue')

    ax[0].set_title('Loss')
    ax[0].set_xlabel('Batch')
    ax[0].set_ylabel('Loss')

    ax[0].legend()

    ax[1].plot(np.array(results['gabornet_correct']) / batch_size, c='orange', alpha=0.2)
    ax[1].plot(np.array(results['cnn_correct']) / batch_size, c='blue', alpha=0.2)

    ax[1].plot(gab_correct, label='gabornet', c='orange')
    ax[1].plot(cnn_correct, label='cnn', c='blue')

    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Batch')
    ax[1].set_ylabel('Accuracy')
    
    ax[1].legend()

    plt.show()


if __name__ == '__main__':
    main()
