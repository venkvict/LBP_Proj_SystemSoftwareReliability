import matplotlib.pyplot as plt
import seaborn as sns

def plot_reliability_growth(times, n_failures):
    plt.figure(figsize=(10, 6))
    plt.step(times, n_failures, where='post')
    plt.title("Software Reliability Growth")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Failures")
    plt.grid(True)
    plt.show()
