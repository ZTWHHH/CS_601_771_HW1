import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os  


class SelfAttention(torch.nn.Module):
    def __init__(self, model_dim):
        super(SelfAttention, self).__init__()
        self.model_dim = model_dim
        self.query = torch.nn.Linear(model_dim, model_dim)
        self.key = torch.nn.Linear(model_dim, model_dim)
        self.value = torch.nn.Linear(model_dim, model_dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_weights = torch.matmul(Q, K.transpose(1, 2)) / (self.model_dim ** 0.5)
        attention_weights = self.softmax(attention_weights)
        out = torch.matmul(attention_weights, V)
        return out


def calculate_flops(seq_len, model_dim):
    flops_proj = 3 * (2 * seq_len * model_dim * model_dim)  
    flops_attention_weights = 2 * seq_len * seq_len * model_dim 
    flops_output = 2 * seq_len * seq_len * model_dim  
    total_flops = flops_proj + flops_attention_weights + flops_output

    return total_flops


def run_experiment(seq_len, model_dim, n_warmup, device):
    input_data = torch.rand(1, seq_len, model_dim).to(device) 
    model = SelfAttention(model_dim).to(device)

    for _ in range(n_warmup):
        _ = model(input_data) 
    
    torch.cuda.reset_peak_memory_stats()  

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()  
    start_event.record()

    output = model(input_data)

    end_event.record()
    torch.cuda.synchronize()

    wall_clock_time = start_event.elapsed_time(end_event) / 1000.0  

    memory = torch.cuda.max_memory_allocated() 

    flops = calculate_flops(seq_len, model_dim)

    return flops, memory, wall_clock_time


def get_data(seq_lengths, model_dim, n_warmup, n_runs, device):
    all_data = []

    for seq_len in seq_lengths:
        for _ in range(n_runs):
            flops, memory, wall_clock_time = run_experiment(seq_len, model_dim, n_warmup, device)

            all_data.append({
                "Sequence Length": seq_len,
                "Metric": "FLOPs",
                "Value": flops
            })

            all_data.append({
                "Sequence Length": seq_len,
                "Metric": "Memory Usage",
                "Value": memory
            })

            all_data.append({
                "Sequence Length": seq_len,
                "Metric": "Wall Clock Time",
                "Value": wall_clock_time
            })
    
    return pd.DataFrame(all_data)  


def plot_result(data, metric, folder_path, file_name):
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 6))

    metric_data = data[data["Metric"] == metric]

    sns.lineplot(x="Sequence Length", y="Value", data=metric_data, marker='o', errorbar='sd')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sequence Length')
    plt.ylabel(metric)
    plt.title(f'{metric} vs Sequence Length')
    plt.grid(True)
    plt.tight_layout()
    
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    plt.show()


def main():
    seq_lengths = [10, 100, 1000, 10000]  
    model_dim = 64
    n_warmup = 10
    n_runs = 100
    device = "cuda"

    folder_path = './gpu_result'
    os.makedirs(folder_path, exist_ok=True)

    data = get_data(seq_lengths, model_dim, n_warmup, n_runs, device)

    plot_result(data, "FLOPs", folder_path, "flops_plot.png")
    plot_result(data, "Memory Usage", folder_path, "memory_plot.png")
    plot_result(data, "Wall Clock Time", folder_path, "time_plot.png")


if __name__ == "__main__":
    main()

