from flask import Flask, request, jsonify
from ML_pipeline import process_smiles

app = Flask(__name__)

#prediction application
@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    smiles = data.get('smiles')

    if not smiles:
        return jsonify({'error': 'SMILES string is required'}), 400
    

    prediction = process_smiles(smiles)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()