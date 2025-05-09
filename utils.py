import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torchaudio.datasets import LibriMix
from sklearn.model_selection import train_test_split


# Define the dataset
class LibriMixDataset:

    def __init__(self, data_dir=r"D:\LibriMix\storage_dir", sample_rate=16000, num_speakers=2, dev_mode=False, train_mode=False, eval_mode=False):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers
        self.dev_mode = dev_mode
        self.eval_mode = eval_mode
        self.train_mode = train_mode

    def load_dataset(self):
        if self.dev_mode:
            train_set = LibriMix(self.data_dir, "dev", sample_rate=self.sample_rate, num_speakers=self.num_speakers)
            train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=42)
            return train_set, val_set
        else:
            train_set = LibriMix(self.data_dir, "train-100", sample_rate=self.sample_rate, num_speakers=self.num_speakers)
            if self.eval_mode:
                test_set = LibriMix(self.data_dir, "test", sample_rate=self.sample_rate, num_speakers=self.num_speakers)
                return train_set, test_set
            else:
                return train_set


class PlotLibriMix:
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = idx

    def plot_graph(self):
        rate, mix, sources = self.dataset.__getitem__(self.idx)
        # Convert tensors to numpy arrays
        mix_np = mix.numpy()
        sources_np = [source.numpy()[0] for source in sources]

        # Define time axis
        time = np.linspace(0, mix_np.shape[1] / rate, num=mix_np.shape[1])

        # Visualize the waveform
        plt.figure(figsize=(12, 10))
        plt.subplot(len(sources) + 1, 1, 1)
        plt.title("Mixed Signal")
        plt.plot(time, mix_np[0])

        for i, source in enumerate(sources_np):
            plt.subplot(len(sources) + 1, 1, i + 2)
            plt.title(f"Source {i+1}")
            plt.plot(time, source)

        plt.tight_layout()
        plt.show()


