# scripts/test_gcs_auth.py

import gcsfs

# Authenticates with your Google account
fs = gcsfs.GCSFileSystem()

# Try listing files in the ROAD++ bucket
files = fs.ls("waymo_open_dataset_road_plus_plus")
print("✅ Files in ROAD++ bucket:", files[:5])
