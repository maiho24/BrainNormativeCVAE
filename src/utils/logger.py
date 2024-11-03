import matplotlib.pyplot as plt
from pathlib import Path


class Logger:
    def __init__(self):
        self.logs = {}
        self.val_logs = {}

    def on_train_init(self, keys):
        for k in keys:
            self.logs[k] = []

    def on_val_init(self, keys):
        for k in keys:
            self.val_logs[k] = []

    def on_train_step(self, logs_dict):
        for k, v in logs_dict.items():
            self.logs[k].append(v)

    def on_val_step(self, logs_dict):
        for k, v in logs_dict.items():
            self.val_logs[k].append(v)


def plot_losses(logger, path, title=''):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))

    # Plot Training Losses
    plt.subplot(1, 2, 1)
    plt.title('Training Loss Values')
    for k, v in logger.logs.items():
        plt.plot(v, label=str(k))
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)

    # Plot Validation Losses
    plt.subplot(1, 2, 2)
    plt.title('Validation Loss Values')
    for k, v in logger.val_logs.items():
        plt.plot(v, label=str(k))
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Average Loss', fontsize=10)
    plt.legend()

    plt.tight_layout()
    save_path = Path(path) / f"Losses{title}.png"
    plt.savefig(save_path)
    plt.close()