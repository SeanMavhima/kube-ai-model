import requests
import json

def test_api():
    url = 'http://localhost:5000/predict'
    
    # Test with image file
    with open('../data/JPEGImages/img_000001.jpg', 'rb') as f:
        files = {'image': f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("🐾 KUBE-AI API Response:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_health():
    response = requests.get('http://localhost:5000/health')
    print("Health check:", response.json())

if __name__ == '__main__':
    test_health()
    test_api()