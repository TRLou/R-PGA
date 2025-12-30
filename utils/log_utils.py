import matplotlib
# Ensure a non-interactive backend for headless training environments.
# Must be set BEFORE importing pyplot.
try:
    matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import List

class TrainingLogger:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.log_file = save_dir / 'training_log.txt'
        self.iteration_losses = []
        self.epoch_stats = []
        
        # Clear log file if it exists
        if self.log_file.exists():
            self.log_file.unlink()

    def log_iteration(self, total_loss, cls_loss, reg_loss):
        self.iteration_losses.append({
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss
        })

    def log_epoch(self, epoch, asr_test, asr_train, ap50_test=float('nan')):
        avg_total_loss = sum(item['total_loss'] for item in self.iteration_losses) / len(self.iteration_losses) if self.iteration_losses else 0
        avg_cls_loss = sum(item['cls_loss'] for item in self.iteration_losses) / len(self.iteration_losses) if self.iteration_losses else 0
        avg_reg_loss = sum(item['reg_loss'] for item in self.iteration_losses) / len(self.iteration_losses) if self.iteration_losses else 0
        
        self.epoch_stats.append({
            'epoch': epoch,
            'asr_test': asr_test,
            'asr_train': asr_train,
            'ap50_test': ap50_test,
            'avg_total_loss': avg_total_loss,
            'avg_cls_loss': avg_cls_loss,
            'avg_reg_loss': avg_reg_loss
        })

        log_message = f"Epoch {epoch}: ASR_Test = {asr_test:.4f}, AP50_Test = {ap50_test:.4f}, ASR_Train = {asr_train:.4f}, Avg Total Loss = {avg_total_loss:.4f}, Avg Cls Loss = {avg_cls_loss:.4f}, Avg Reg Loss = {avg_reg_loss:.4f}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
            
        # Clear iteration losses for the next epoch
        self.iteration_losses = []

    def plot_iteration_losses(self):
        # This might not be very useful if iterations are per-camera, but keeping it
        if not self.iteration_losses:
            return
        
        total_losses = [item['total_loss'] for item in self.iteration_losses]
        cls_losses = [item['cls_loss'] for item in self.iteration_losses]
        reg_losses = [item['reg_loss'] for item in self.iteration_losses]
        
        plt.figure(figsize=(10, 6))
        plt.plot(total_losses, label='Total Loss')
        plt.plot(cls_losses, label='Classification Loss')
        plt.plot(reg_losses, label='Regression Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss per Iteration (Last Epoch)')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / 'iteration_loss_curve.png')
        plt.close()

    def plot_epoch_losses(self):
        if not self.epoch_stats:
            return
        
        epochs = [item['epoch'] for item in self.epoch_stats]
        avg_total_losses = [item['avg_total_loss'] for item in self.epoch_stats]
        avg_cls_losses = [item['avg_cls_loss'] for item in self.epoch_stats]
        avg_reg_losses = [item['avg_reg_loss'] for item in self.epoch_stats]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, avg_total_losses, label='Avg Total Loss')
        plt.plot(epochs, avg_cls_losses, label='Avg Classification Loss')
        plt.plot(epochs, avg_reg_losses, label='Avg Regression Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Average Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / 'epoch_loss_curve.png')
        plt.close()

    def plot_ap_curve(self):
        if not self.epoch_stats:
            return

        epochs = [item['epoch'] for item in self.epoch_stats]
        ap50s = [item['ap50_test'] for item in self.epoch_stats]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, ap50s, marker='o', linestyle='-', label='AP@0.5 (Test)')
        plt.xlabel('Epoch')
        plt.ylabel('AP@0.5')
        plt.title('Average Precision (AP@0.5) per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / 'ap50_curve.png')
        plt.close()

    def plot_asr_and_loss(self):
        if not self.epoch_stats:
            return

        epochs = [item['epoch'] for item in self.epoch_stats]
        asrs_test = [item['asr_test'] for item in self.epoch_stats]
        asrs_train = [item['asr_train'] for item in self.epoch_stats]
        avg_total_losses = [item['avg_total_loss'] for item in self.epoch_stats]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('ASR', color=color)
        ax1.plot(epochs, asrs_test, color=color, marker='o', linestyle='-', label='ASR (Test)')
        ax1.plot(epochs, asrs_train, color='tab:green', marker='s', linestyle='--', label='ASR (Train)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Average Total Loss', color=color)
        ax2.plot(epochs, avg_total_losses, color=color, marker='x', linestyle=':', label='Avg Total Loss')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('ASR and Average Loss vs. Epochs')
        
        # Add legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')

        plt.savefig(self.save_dir / 'asr_vs_loss_curve.png')
        plt.close()
