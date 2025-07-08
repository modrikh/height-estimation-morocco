import matplotlib.pyplot as plt

def plot_training_history(history):
    metrics = ['loss', 'rmse_metric', 'mae_metric', 'r2_metric']

    for metric in metrics:
        plt.figure(figsize=(6, 4))
        plt.plot(history.history[metric], label=f'Train {metric}')
        val_key = f'val_{metric}'
        if val_key in history.history:
            plt.plot(history.history[val_key], label=f'Val {metric}')
        plt.title(f'{metric} over epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
