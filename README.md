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

The recorded data will be:
1. Audio recordings
    - 30 second microphone recordings
    - split into 5 10 second chunks (5 seconds of overlap)
2. Temperature (internal)
3. Atmospheric pressure
4. relative humidity (internal)
5. total hive weight

Machine learning data row:
```
(sample_id, hive_id,
chunk_start, chunk_end,
audio_path, 
"sensor data (varies)",
labels
```
