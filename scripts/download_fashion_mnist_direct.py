import gzip
import urllib.request
from pathlib import Path

import numpy as np


def fetch_fashion_mnist_direct(data_dir: Path):
    base_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    arrays = []
    for f in files:
        url = base_url + f
        path = data_dir / f
        if not path.exists():
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, path)
        
        with gzip.open(path, 'rb') as gz:
            if 'images' in f:
                # Skip magic number (4) and counts (12)
                gz.read(16)
                buf = gz.read()
                data = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)
                arrays.append(data.reshape(-1, 784))
            else:
                # Skip magic number (4) and count (4)
                gz.read(8)
                buf = gz.read()
                data = np.frombuffer(buf, dtype=np.uint8)
                arrays.append(data)
                
    X_train, y_train, X_test, y_test = arrays
    X = np.concatenate([X_train, X_test], axis=0).astype(np.float64)
    y = np.concatenate([y_train, y_test], axis=0).astype(np.int64)
    
    return X, y
