import requests
import json

config = json.loads(open('../config/api_key.json').read())
key = config['api_key']
enfpoint = config['endpoint']

