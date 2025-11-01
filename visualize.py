import matplotlib.pyplot as plt
import csv
from io import StringIO

OUTPUT_DIR = "results/"

def plot_initial_dense_train_accuracy_over_epochs(out_name, data_content):
    """
    Parses data from a CSV-like string content and plots the training and test accuracy.
    Assumes the columns are: Epoch, Loss, Train Acc, Test Acc.
    """
    epochs = []
    train_acc = []
    test_acc = []

    # Use StringIO to treat the string content like a file
    csvfile = StringIO(data_content)
    reader = csv.reader(csvfile, skipinitialspace=True)
    
    # Skip header
    next(reader)

    for row in reader:
        epoch = int(row[0].split('/')[0])
        t_acc = float(row[2])
        ts_acc = float(row[3])
        
        epochs.append(epoch)
        train_acc.append(t_acc)
        test_acc.append(ts_acc)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Training Accuracy', marker='o', linestyle='-', linewidth=2)
    plt.plot(epochs, test_acc, label='Validation/Test Accuracy', marker='x', linestyle='--', linewidth=2)
    plt.title('AlexNet Accuracy Over Training Epochs (CIFAR-10)', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)
    plt.ylim(min(train_acc + test_acc) * 0.95, 100) # Ensure y-axis starts near the data
    max_epoch = max(epochs)
    tick_interval = 50
    tick_marks = list(range(0, max_epoch + tick_interval, tick_interval))
    if max_epoch not in tick_marks:
        tick_marks.append(max_epoch)
        tick_marks.sort()
    plt.xticks(tick_marks)
    
    # Add a conversational label for the divergence point
    # We use a placeholder for now, but this is where you'd analyze overfitting
    plt.text(epochs[-1] - 5, test_acc[-1] - 1, 'Test Accuracy Plateau', 
             fontsize=9, color='red', horizontalalignment='right')
    
    plt.tight_layout()
    plt.savefig(out_name)
    print(f"Plot successfully saved to {out_name}")
    plt.show() # Display the plot


if __name__ == "__main__":
    with open(OUTPUT_DIR + 'initAlexNetCIFAR10avg.csv', 'r') as f:
        data_content = f.read()
        plot_initial_dense_train_accuracy_over_epochs(OUTPUT_DIR + "plots/initAlexNetCIFAR10avg.png", data_content)
