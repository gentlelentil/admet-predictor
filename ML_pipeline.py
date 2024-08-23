#import libraries
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch.utils.data import Dataset, DataLoader, Subset, random_split


import joblib
import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch import nn
import sys
import json

from rdkit.Chem import AllChem as Chem, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
desc_calc = MolecularDescriptorCalculator([x for x in [x[0] for x in Descriptors.descList]])

#placeholder for descriptors
# molecular_descriptor_list = []

#define model architecture

class Linear_3L(torch.nn.Module):
    def __init__(self,input_dim, *args, **kwargs) -> None:
        super(Linear_3L, self).__init__(*args, **kwargs)
        
        self.Lin0 = torch.nn.Linear(input_dim, 2000)
        self.batchnorm1 = torch.nn.BatchNorm1d(2000)
        self.dropout = torch.nn.Dropout(0.75)

        self.Lin1 = torch.nn.Linear(2000, 500)
        self.batchnorm2 = torch.nn.BatchNorm1d(500)
        self.Lin2 = torch.nn.Linear(500, 10)
        self.batchnorm3 = torch.nn.BatchNorm1d(10)

        self.linout = Linear(10, 1)

    def forward(self, x):
        # x = data.x

        #L1
        out = F.relu(self.Lin0(x))
        out = F.relu(self.batchnorm1(out))
        out = self.dropout(out)
        # print('L1')

        out = F.relu(self.Lin1(out))
        out = F.relu(self.batchnorm2(out))
        out = self.dropout(out)
        # print('L2')

        out = F.relu(self.Lin2(out))
        out = F.relu(self.batchnorm3(out))
        out = self.dropout(out)
        # print('L3')

        out = torch.sigmoid(self.linout(out))
        out = out.view(-1)

        return out

model_names = ['oral_abs_class',
 'hia_class',
 'crl_toxicity_class',
 'ML_input_p450-cyp3a4',
 'ames_mutagenicity_class',
 'ML_input_p450-cyp2c19',
 'hep_g2_toxicity_class',
 'nih_toxicity_class',
 'herg_blockers_class',
 'hek_toxicity_class',
 'hacat_toxicity_class',
 'ML_input_p450-cyp1a2',
 'bbb_class',
 'ML_input_p450-cyp2d6',
 'ML_input_p450-cyp2c9',]


preprocessors = {}
models = {}

for model_name in model_names:
    #load preprocessors
    imputer = joblib.load(f'./data/{model_name}/{model_name}_imputer.pkl')
    scaler = joblib.load(f'./data/{model_name}/{model_name}_scaler.pkl')
    selected_features = joblib.load(f'./data/{model_name}/{model_name}_selected_features.pkl')
    

    preprocessors[model_name] = {
        'imputer': imputer,
        'scaler': scaler,
        'selected_features': selected_features
    }

    #load model
    input_dim = len(selected_features)
    model = Linear_3L(input_dim)
    model.load_state_dict(torch.load(f'./MODELS/{model_name}_statedict.pt', map_location=torch.device('cpu'), weights_only=True))
    # model.load_state_dict(torch.load(f'./MODELS/{model_name}_statedict.pt'), map_location=torch.device('cpu'))
    model.eval()

    models[model_name] = model

#preprocess_smiles
def preprocess_smiles(smiles: str, model_name: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES')

    descriptor = np.array(desc_calc.CalcDescriptors(mol)).reshape(1, -1)

    #apply preprocessing

    imputer = preprocessors[model_name]['imputer']
    scaler = preprocessors[model_name]['scaler']
    selected_features = preprocessors[model_name]['selected_features']

    descriptor = imputer.transform(descriptor)
    descriptor = scaler.transform(descriptor)
    descriptor = descriptor[:, selected_features]

    return torch.tensor(descriptor, dtype=torch.float32)

#loop to receive SMILES input and run inference
def process_smiles(smiles):
    results = {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES string"}
    
    for model_name in model_names:
        try:
            # Preprocess the SMILES for the current model
            input_data = preprocess_smiles(smiles, model_name)

            if input_data is None:
                results[model_name] = "Error: Invalid SMILES"
                continue

            # Run the model and make a prediction
            with torch.no_grad():
                prediction = models[model_name](input_data)

            # Format and output the result
            results[model_name] = f"{prediction.item():.4f}"

        except Exception as e:
            results[model_name] = f"Error: {str(e)}"
    
    return results

# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print("Usage: python script.py <SMILES>")
#         sys.exit(1)

#     smiles_input = sys.argv[1]
#     results = process_smiles(smiles_input)
#     print(json.dumps(results, indent=4))
    # process_smiles()