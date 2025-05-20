import os
import subprocess
import tarfile
from urllib.request import urlretrieve
import logging
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

def main():
    # Check if models are already downloaded in ./models directory
    if not os.path.exists("./models"):
        logger.info("No models found. Downloading models...")
        download_models()
        logger.info("Models downloaded.")
    else:
        logger.info("Models already downloaded.")

def download_models():
    # Download the file
    url = "https://gitlab.com/davidblanar/nutrition5k/-/archive/main/nutrition5k-main.tar.gz?path=nutrition_model/models"
    output_path = "/models/models.tar.gz"
    os.makedirs("/models", exist_ok=True)
    urlretrieve(url, output_path)

    # List files in /models directory
    print("\nFiles in /models directory:")
    print(os.listdir("/models"))

    # Extract the tar.gz file
    with tarfile.open(output_path, "r:gz") as tar:
        tar.extractall(path="/models", members=strip_components(tar, 3))

def strip_components(tar, strip_count):
    for member in tar.getmembers():
        path_parts = member.name.split("/")
        member.name = "/".join(path_parts[strip_count:])
        yield member

if __name__ == "__main__":
    main()
