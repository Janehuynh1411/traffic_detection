import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# scripts/load_annotations.py

import gcsfs
import json
from data_config import GCS_ANNOTATION_PATH

# Create a filesystem interface to Google Cloud Storage
fs = gcsfs.GCSFileSystem(token='anon')

# Load the annotation JSON from Google Cloud
with fs.open(GCS_ANNOTATION_PATH, 'r') as f:
    data = json.load(f)

# Show how many videos are annotated
print("✅ Loaded annotations!")
print("Total videos:", len(data['db']))

# Show one video as a sample
sample_video_id = list(data['db'].keys())[0]
print("Example video ID:", sample_video_id)
print("Fields in this video's annotation:", list(data['db'][sample_video_id].keys()))
