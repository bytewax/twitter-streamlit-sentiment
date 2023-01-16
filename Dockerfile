FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install --default-timeout=900 transformers[torch]==4.21.2
RUN pip3 install -r requirements.txt

RUN python3 -m spacy download en_core_web_sm

# Load the BERT model from Huggingface and store it in the model directory
RUN mkdir model
RUN curl -L https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest/resolve/main/pytorch_model.bin -o ./model/pytorch_model.bin
RUN curl https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest/raw/main/config.json -o ./model/config.json
RUN curl https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest/raw/main/merges.txt -o ./model/merges.txt
RUN curl https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest/raw/main/special_tokens_map.json -o ./model/special_tokens_map.json
RUN curl https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest/raw/main/vocab.json -o ./model/vocab.json



ENTRYPOINT ["streamlit", "run", "dataflow.py", "--server.port=8501", "--server.address=0.0.0.0"]