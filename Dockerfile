# Use Python 3.11 as the base image
FROM python:3.11-slim

# Install necessary dependencies (curl and tar)
RUN apt-get update && apt-get install -y \
    curl \
    tar \
    && rm -rf /var/lib/apt/lists/*

ENV STREAMLIT_SERVER_HEADLESS=true

# Install Ollama inside the container (use the appropriate method)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Ensure Ollama is in the PATH
ENV PATH="/root/.ollama/bin:${PATH}"

RUN ollama --version

# Pull the llama3.2 model (downloads the model)
# RUN ollama pull llama3.2

# Set the working directory
WORKDIR /app

# Copy application files
COPY . .

RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add local LLaMA model and Chroma DB as volumes
VOLUME ["/models/llama", "/db/chroma"]

# Expose the necessary ports (e.g., Streamlit runs on 8501 by default)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "chatbot.py", "--server.port=8501"]

