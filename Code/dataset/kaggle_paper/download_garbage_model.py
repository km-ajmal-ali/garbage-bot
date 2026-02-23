import kagglehub

# Download latest version
path = kagglehub.dataset_download("spellsharp/garbage-data")

print("Path to dataset files:", path)