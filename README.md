## ADMET predictor.

Model trained on 15 ADMET Datasets, input SMILES sequence and output model predictions

Test prediction if running:

curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"smiles": "C1=CC=CC=C1"}'