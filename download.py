# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from min_dalle import MinDalle
import torch


def download_model():
    model = MinDalle(
        models_root='./pretrained',
        dtype=torch.float32,
        device='cpu',
        is_mega=True,
        is_reusable=True
    )

if __name__ == "__main__":
    download_model()