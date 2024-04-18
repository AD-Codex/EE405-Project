import requests
import shutil

# URL of the Google Drive photo
google_drive_url = "https://drive.google.com/file/d/1YGT2lnEVeK4JGVH6ccZeyuLgM9i6IGzM/view?usp=sharing"

# Filename to save the photo as
filename = "downloaded_photo.jpg"

# Send a GET request to the Google Drive URL
response = requests.get(google_drive_url, stream=True)

# Check if the request was successful
if response.status_code == 200:
    # Open a local file to save the photo
    with open(filename, "wb") as file:
        # Copy the content of the response to the file
        shutil.copyfileobj(response.raw, file)
    
    print(f"Photo downloaded and saved as '{filename}'.")
else:
    print("Failed to download the photo.")