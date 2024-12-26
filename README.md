# Doc Chatbot

A chatbot application that allows users to upload PDF documents, process them, and ask questions about the document's content. Built using **LangChain** and **Ollama**, this application is containerized using **Docker** for easy deployment.

## Prerequisites

Before you can run the Docker container, make sure you have the following installed on your machine:

- **[Docker](https://docs.docker.com/get-docker/)**: Follow the installation instructions for your platform. Leave it running in the background. You possible need to add the path to your ~/.zshrc.
- **[Docker Compose](https://docs.docker.com/compose/install/)** (optional but recommended): For managing multi-container Docker applications.

## Setup Instructions

### 1. Clone the Repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/JardennaM/doc_chatbot.git
cd doc_chatbot
```

### 2. Build the app

```bash
docker-compose build
docker-compose up
```

### 3. Access the application
Paste the following in your browser:
```bash
http://localhost:8501
```

