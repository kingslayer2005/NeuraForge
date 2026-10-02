import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.datasets import fetch_and_cache_openml

def download_fashion_mnist_robust():
    for attempt in range(5):
        try:
            print(f"Attempt {attempt+1} to download Fashion-MNIST...")
            fetch_and_cache_openml("Fashion-MNIST", version=1, cache_name="fashion_mnist")
            print("Successfully downloaded and cached!")
            return
        except Exception as e:
            print(f"Failed: {e}")
            time.sleep(2)

if __name__ == "__main__":
    download_fashion_mnist_robust()
