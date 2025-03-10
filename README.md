# Soorat Model Inference
This project is designed to enable model inference

# How to setup

## Setup the environment (Python 3.9)

### Create a virtual environment
```bash
python -m venv venv
```
### Activate the virtual environment
```bash
source venv/bin/activate
```

## Setup the environment (Conda)

### Create a conda environment
```bash
conda create --name soorat python=3.9
```
### Activate the conda environment
```bash
conda activate soorat
```

## Install the requirements

### Install the requirements
```bash
pip install -r requirements.txt
```

### Install the requirements (Conda)
```bash
conda install --file requirements.txt
```

## Setup the Project

### Add the model
Add the models to `/models` folder.

Download the models from:
[GitLab | Zdavidblanar/nutrition5k](https://gitlab.com/davidblanar/nutrition5k/-/tree/main/nutrition_model/models?ref_type=heads)


# Run the project
```bash
fastapi run main.py
```

## WSL find the IP address
```bash
ip addr show eth0 | grep inet | awk '{ print $2; }' | sed 's/\/.*$//'
```

## Access the API docs
In the API docs you'll find the endpoints and documentation on how to use them.
```bash
http://IP_ADDRESS:PORT/docs
```
