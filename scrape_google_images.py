from google_images_download import google_images_download
from pathlib import Path


search_terms = {
    "Cassava Bacterial Blight": "cbb-0",
    "Cassava CBB": "cbb-1",
    "Cassava Brown Streak Disease": "cbsd-0",
    "Cassava CBSD": "cbsd-1",
    "Cassava Green Mottle": "cgm-0",
    "Cassava CGM": "cgm-1",
    "Cassava Mosaic Disease": "cmd-0",
    "Cassava CMD": "cmd-1",
    "cassava plant": "healthy-0",
    "cassava leaves": "healthy-1"
}

download_dir = Path('/mnt/hdd/leaf-disease-data/images/')
download_dir.mkdir(exist_ok=True)

for search_term, dir_prefix in search_terms.items():
    (download_dir / dir_prefix).mkdir(exist_ok=True)
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords":search_term,
                 #'format':'jpg',
                 #'size':'medium',
                 "limit":1000,
                 "output_directory":str(download_dir / dir_prefix),
                 "no_directory":True,
                 "chromedriver":"./chromedriver"}
    paths = response.download(arguments)
