"""
download_data.py
Downloads and prepares evaluation datasets from common sources.
"""

import os
import requests
import gdown
import zipfile
from tqdm import tqdm
from pathlib import Path

class DataDownloader:
    def __init__(self):
        self.base_dir = Path("data")
        self.word_sets_dir = self.base_dir / "word_sets"
        self.models_dir = self.base_dir / "models"
        
        # Create directories
        self.word_sets_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, dest_path: Path, desc: str = "Downloading"):
        """Download a file with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

    def download_word_sets(self):
        """Download word sets from various bias evaluation papers."""
        
        # WEAT word sets from Caliskan et al.
        weat_url = "https://raw.githubusercontent.com/w4ngatang/sent-bias/master/data/weat.json"
        self.download_file(weat_url, self.word_sets_dir / "weat.json", "Downloading WEAT word sets")

        # Professional word sets from Bolukbasi et al.
        professions_url = "https://raw.githubusercontent.com/tolga-b/debiaswe/master/data/professions.json"
        self.download_file(professions_url, self.word_sets_dir / "professions.json", "Downloading profession word sets")

    def download_word_embeddings(self):
        """Download pre-trained word embeddings."""
        # Download Google's word2vec embeddings
        word2vec_url = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
        word2vec_path = self.models_dir / "GoogleNews-vectors-negative300.bin.gz"
        
        if not word2vec_path.exists():
            print("Downloading word2vec embeddings (this might take a while)...")
            gdown.download(word2vec_url, str(word2vec_path), quiet=False)
        
        # Note: For GloVe embeddings, users need to download manually due to license requirements
        print("\nNote: For GloVe embeddings, please download manually from:")
        print("https://nlp.stanford.edu/data/glove.840B.300d.zip")

def main():
    downloader = DataDownloader()
    
    print("Starting data download...")
    
    # Download word sets
    print("\nDownloading word sets...")
    downloader.download_word_sets()
    
    # Download word embeddings
    print("\nDownloading word embeddings...")
    downloader.download_word_embeddings()
    
    print("\nDownload complete! Directory structure:")
    print("\ndata/")
    print("├── word_sets/")
    print("│   ├── weat.json")
    print("│   └── professions.json")
    print("└── models/")
    print("    └── GoogleNews-vectors-negative300.bin.gz")

if __name__ == "__main__":
    main()