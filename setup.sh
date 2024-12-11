#!/bin/bash

# Make setup.sh executable
chmod +x setup.sh

# Upgrade pip
pip install --upgrade pip

# Install spacy model
python -m spacy download en_core_web_sm

# Create streamlit config directory
mkdir -p ~/.streamlit/

# Create streamlit config file
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
