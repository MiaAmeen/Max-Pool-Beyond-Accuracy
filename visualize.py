import matplotlib.pyplot as plt
import csv
import numpy as np

OUTPUT_DIR = "results/"
AVG_POOL_RESULTS_FILE = OUTPUT_DIR + 'initAlexNetCIFAR10avg.csv'
MAX_POOL_RESULTS_FILE = OUTPUT_DIR + 'initAlexNetCIFAR10max.csv'

# Max: best performing seeds are 14, 10, 3, 2, 0, average accuracy ~ 86%
# Avg: best performing seeds are 16, 5, 8, 4, 2,  average accuracy = 85.072%
def plot_initial_dense_train_accuracy_over_epochs(out_name,
                                                  avgpoolresultsfile=AVG_POOL_RESULTS_FILE,
                                                  maxpoolresultsfile=MAX_POOL_RESULTS_FILE):

    def load_results(path):
        train_accs = {}
        test_accs = {}
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if int(row[0]) > 15: continue
                epoch = int(row[1].split('/')[0])
                train_acc = float(row[3])
                test_acc = float(row[4])
                if epoch not in train_accs:
                    train_accs[epoch] = []
                    test_accs[epoch] = []
                train_accs[epoch].append(train_acc)
                test_accs[epoch].append(test_acc)
        return train_accs, test_accs

    avg_train, avg_test = load_results(avgpoolresultsfile)
    max_train, max_test = load_results(maxpoolresultsfile)
    epochs = [i for i in range(1, 101)]

    def compute_mean_sem(acc_dict):
        means = []
        sems = []
        for epoch in epochs:
            vals = np.array(acc_dict[epoch])
            means.append(vals.mean())
            sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))
        return np.array(means), np.array(sems)

    avg_train_mean, avg_train_sem = compute_mean_sem(avg_train)
    avg_test_mean, avg_test_sem = compute_mean_sem(avg_test)
    max_train_mean, max_train_sem = compute_mean_sem(max_train)
    max_test_mean, max_test_sem = compute_mean_sem(max_test)


    plt.figure(figsize=(11, 7))
    avg_color = "tab:blue"
    max_color = "tab:orange"

    # --- AvgPool ---
    plt.plot(epochs, avg_test_mean, label="AvgPool – Test", color=avg_color, linewidth=2)
    plt.fill_between(epochs,
                     avg_test_mean - avg_test_sem,
                     avg_test_mean + avg_test_sem,
                     color=avg_color,
                     alpha=0.5)

    plt.plot(epochs, avg_train_mean, label="AvgPool – Train", color=avg_color, linestyle="--", linewidth=2)
    plt.fill_between(epochs,
                     avg_train_mean - avg_train_sem,
                     avg_train_mean + avg_train_sem,
                     color=avg_color,
                     alpha=0.5)

    # --- MaxPool ---
    plt.plot(epochs, max_test_mean, label="MaxPool – Test", color=max_color, linewidth=2)
    plt.fill_between(epochs,
                     max_test_mean - max_test_sem,
                     max_test_mean + max_test_sem,
                     color=max_color,
                     alpha=0.5)

    plt.plot(epochs, max_train_mean, label="MaxPool – Train", color=max_color, linestyle="--", linewidth=2)
    plt.fill_between(epochs,
                     max_train_mean - max_train_sem,
                     max_train_mean + max_train_sem,
                     color=max_color,
                     alpha=0.5)

    # --- Formatting ---
    plt.title("AlexNet CIFAR-10 Accuracy\nMean ± Standard Error Across 15 Seeds", fontsize=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_name)
    plt.show()

    print(f"Saved plot → {out_name}")


if __name__ == "__main__":
    plot_initial_dense_train_accuracy_over_epochs(
        OUTPUT_DIR + "plots/initAlexNetCIFAR10avgVSmax.png"
    )
