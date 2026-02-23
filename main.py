import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem
from fastapi.middleware.cors import CORSMiddleware

# 1. Initialize API
app = FastAPI()

# Enable CORS for your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Advanced Architecture (Must match your 84.48% Kaggle model)
class Advanced_DDI_Predictor(nn.Module):
    def __init__(self, input_size=2048, num_classes=4):
        super(Advanced_DDI_Predictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# 3. Load AI Brain (Using CPU for Render Free Tier)
try:
    le = joblib.load('Severity_LabelEncoder.pkl')
    model = Advanced_DDI_Predictor(num_classes=len(le.classes_))
    model.load_state_dict(torch.load('DDInter_DeepLearning_Model.pth', map_location=torch.device('cpu')))
    model.eval()
    print("✅ AI Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

class DrugInput(BaseModel):
    smiles_a: str
    smiles_b: str

def get_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

# 4. Prediction Endpoint
@app.post("/predict")
async def predict(data: DrugInput):
    try:
        fp_a = get_fp(data.smiles_a)
        fp_b = get_fp(data.smiles_b)
        
        if fp_a is None or fp_b is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES string.")

        # Prepare Input
        x_input = np.concatenate([fp_a, fp_b])
        x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(x_tensor)
            # Calculate Confidence via Softmax
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            conf, pred_idx = torch.max(probs, 0)

        return {
            "status": "success",
            "severity": le.inverse_transform([pred_idx.item()])[0],
            "confidence": round(conf.item() * 100, 2),
            "drug_a": data.smiles_a,
            "drug_b": data.smiles_b
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check for Render
@app.get("/")
def home():
    return {"message": "MedCare DDI API is live!"}