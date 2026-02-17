Rutgers Capstone '26 Hive Monitoring System AI Audio+Telemetry Classifier Project

Plan: 
Use Autio Spectrogram Transformer as a feature extractor to create embeddings of beehive audio recordings. 
Feed embeddings into small classifer head to monitor the internal state of a beehive.
Incorporate other telemetry measurements in state classification (temperature, humidity, pressure, weight)

Project Structure
```
hms-inference/
    pyproject.toml
    README.md
    requirements.txt
    scripts/
        embed_one.py
    src/
        hms_inference/
            ast_embedder.py
            audio_io.py
            __init__.py   
        data/
            UrBAN/
``` 
