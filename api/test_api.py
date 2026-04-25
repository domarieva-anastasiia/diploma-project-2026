import requests

url = "http://127.0.0.1:5000/enhance"

with open("api/test_1.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

print("Status:", response.status_code)

if response.status_code == 200:
    with open("result.png", "wb") as out:
        out.write(response.content)
    print("Saved result.png")
else:
    print("Error:", response.text)