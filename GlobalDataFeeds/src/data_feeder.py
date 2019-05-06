import json
from time import time

from flask import Flask, jsonify, request
import requests

# config = json.parse('../config/api_key.json')

class DataFeeder(config):
    def __init__(self, config):
        self.config = config['api_key']
        self.endpoint = config['endpoint']
        
    def writer(self, response):
        pass
    
    def get_response(self, **kwargs):
        pass
    
    
# Instantiate our Node



@app.route('/health', methods=['GET'])
def full_chain():
    response = {
        'status': 'healthy'
    }
    return jsonify(response), 200

@app.route('/get_data', methods=['GET'])
def get_data():
    # We run the proof of work algorithm to get the next proof...
    last_block = blockchain.last_block
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)

    # We must receive a reward for finding the proof.
    # The sender is "0" to signify that this node has mined a new coin.
    blockchain.new_transaction(
        sender="0",
        recipient=node_identifier,
        amount=1,
    )

    # Forge the new Block by adding it to the chain
    previous_hash = blockchain.hash(last_block)
    block = blockchain.new_block(proof, previous_hash)

    response = {
        'message': "New Block Forged",
        'index': block['index'],
        'transactions': block['transactions'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200

if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
    config = json.load(open('config/api_key.json'))
    
    print(config)
    app = Flask(__name__)

    feeder = DataFeeder()   
#     feeder = DataFeeder(config)
    print(feeder.config)
    
    
    
    
    