import requests

url = 'https://v3.exchangerate-api.com/bulk/2c4f52b308a96a049fa5a275/USD'

response = requests.get(url)
data = response.json()

data
data['rates']['GBP']
