import logging
from datetime import datetime
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
            value = float(v) if hasattr(v, 'item') else v
            self.logs[k].append(v)

    def on_val_step(self, logs_dict):
        for k, v in logs_dict.items():
            value = float(v) if hasattr(v, 'item') else v
            self.val_logs[k].append(v)


def plot_losses(logger, path, title=''):
    plt.figure(figsize=(10, 6))
    
    # Find the global min and max across all losses for consistent y-axis
    all_values = []
    for v in logger.logs.values():
        all_values.extend(v)
    for v in logger.val_logs.values():
        all_values.extend(v)
    
    if all_values:
        y_min = min(all_values) - 5
        y_max = max(all_values) + 5
    else:
        y_min, y_max = 0, 1
        
    # Plot Training Losses
    plt.subplot(1, 2, 1)
    plt.title('Training Loss Values')
    for k, v in logger.logs.items():
        plt.plot(v, label=str(k))
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Average Loss', fontsize=10)
    plt.ylim(y_min, y_max)
    
    # Plot Validation Losses  
    plt.subplot(1, 2, 2)
    plt.title('Validation Loss Values')
    for k, v in logger.val_logs.items():
        plt.plot(v, label=str(k))
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Average Loss', fontsize=10)
    plt.ylim(y_min, y_max)
    plt.legend()
    
    plt.tight_layout()
    save_path = Path(path) / f"Losses{title}.png"
    plt.savefig(save_path)
    plt.close()
    
def setup_logging(output_dir, script_name):
    """Set up logging with timestamp in filename."""
    # Create logs directory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{script_name}_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file))
        ]
    )
    return logging.getLogger(__name__)