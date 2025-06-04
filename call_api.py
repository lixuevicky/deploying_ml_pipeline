import requests

url = "https://auto-deploy-ml-60a8acf1b017.herokuapp.com/inference"

payload = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

response = requests.post(url, json=payload)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")