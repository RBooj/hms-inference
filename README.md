Rutgers Capstone '26 Hive Monitoring System AI Audio+Telemetry Classifier Project

Plan: 
Use Autio Spectrogram Transformer as a feature extractor to create embeddings of beehive audio recordings. 
Feed embeddings into small classifer head to monitor the internal state of a beehive.
Incorporate other telemetry measurements in state classification (temperature, humidity, pressure, weight)

Project Structure
```hms-inference/
    README.md
    requirements.txt
    src/
        hms-audio/
            __init__.py
            config.py
            audio_io.py
            ast_embedder.py
        scripts/
            embed_one.py
        data/
            UrBAN/
```

Github ignores:
    
