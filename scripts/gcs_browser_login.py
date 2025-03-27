# scripts/gcs_browser_login.py

import gcsfs

# 🔑 This will open your browser and ask you to log in
fs = gcsfs.GCSFileSystem(token='browser')

# Try listing files from the ROAD++ bucket
files = fs.ls("waymo_open_dataset_road_plus_plus")
print("✅ ACCESS GRANTED! First 5 files in bucket:")
print(files[:5])
