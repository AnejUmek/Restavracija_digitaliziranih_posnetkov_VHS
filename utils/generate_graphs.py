from pathlib import Path
from matplotlib import pyplot as plt
import json

def generate_graphs_train(data_log_path: Path) -> None:
    graphs_path = create_folder(data_log_path)

    with open(str(data_log_path), "r") as json_file:
        data = json.load(json_file)

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        
        for i, train_loss_name in enumerate(data["train"].keys()):
            ax[i].plot(data["train"][train_loss_name])
            ax[i].set_title(str(train_loss_name).replace("_", " ").capitalize())
            ax[i].grid(True)
        ax[-1].set_xlabel("epoch")
        plt.tight_layout()
        plt.savefig(f"{str(graphs_path / 'train_losses')}.png")
        plt.close(fig)

def generate_graphs_validation(data_log_path: Path) -> None:
        graphs_path = create_folder(data_log_path)
        
        with open(str(data_log_path), "r") as json_file:
            data = json.load(json_file)
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

            for i, validation_metric_name in enumerate(data["validation"].keys()):
                ax[i].plot(data["validation"][validation_metric_name])
                ax[i].set_title(str(validation_metric_name).upper())
                ax[i].grid(True)
            ax[-1].set_xlabel("epoch")
            plt.tight_layout()
            plt.savefig(f"{str(graphs_path / 'validation_metrics')}.png")
            plt.close(fig)

def generate_graphs_test(data_log_path: Path) -> None:
    
    graphs_path = create_folder(data_log_path)
        
    with open(str(data_log_path), "r") as json_file:
        data = json.load(json_file)
        
        for video_name in data["test"].keys():
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
            for i, test_metric_name in enumerate(data["test"][video_name]["restored"].keys()):
                ax[i].plot(data["test"][video_name]["restored"][test_metric_name], label="Restored")
                ax[i].plot(data["test"][video_name]["input"][test_metric_name], label="Input")
                ax[i].set_title(str(test_metric_name).upper())
                ax[i].grid(True)
                ax[i].legend()
            ax[-1].set_xlabel("frame")
            fig.suptitle(video_name, fontsize=16)
            plt.tight_layout()
            file_name = f"{video_name}_test_metrics"
            plt.savefig(f"{str(graphs_path / file_name)}.png")
            plt.close(fig)
        
def create_folder(data_log_path: Path) -> Path:
    data_log_path_parent = data_log_path.parent
    graphs_path = data_log_path_parent / "graphs"
    graphs_path.mkdir(parents=False, exist_ok=True)
    return graphs_path