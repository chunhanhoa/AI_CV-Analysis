#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install core dependencies first
pip install numpy==1.26.2
pip install pandas==2.1.4

# Install spacy and download model
pip install spacy==3.7.2
python -m spacy download en_core_web_sm

# Create streamlit config directory
mkdir -p ~/.streamlit/

# Create streamlit config file
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = \$PORT\n\
" > ~/.streamlit/config.toml
