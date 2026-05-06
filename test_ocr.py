import requests

url = "http://localhost:8000/api/v1/ocr/extract"

with open(r"D:\nlp-pipeline\Ima.jpg", "rb") as f:
    response = requests.post(url, files={"file": ("Ima.jpg", f, "image/jpeg")})

print(response.json())