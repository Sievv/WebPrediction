import numpy as np
import joblib
import streamlit as st
from Bio.SeqUtils import molecular_weight
from Bio.Seq import Seq

# ======================
# 1. Load all models
# ======================
@st.cache_resource
def load_models():
    return {
        "PeptideP": {
            "RandomForest":     joblib.load("PeptideP_RandomForest.joblib"),
            "GradientBoosting": joblib.load("PeptideP_GradientBoosting.joblib"),
            "AdaBoost":         joblib.load("PeptideP_AdaBoost.joblib"),
            "SVM":              joblib.load("PeptideP_SVM.joblib"),
            "XGBoost":          joblib.load("PeptideP_XGBoost.joblib"),
        },
        "PeptideE": {
            "RandomForest":     joblib.load("PeptideE_RandomForest.joblib"),
            "GradientBoosting": joblib.load("PeptideE_GradientBoosting.joblib"),
            "AdaBoost":         joblib.load("PeptideE_AdaBoost.joblib"),
            "SVM":              joblib.load("PeptideE_SVM.joblib"),
            "XGBoost":          joblib.load("PeptideE_XGBoost.joblib"),
        },
        "PeptideK": {
            "RandomForest":     joblib.load("PeptideK_RandomForest.joblib"),
            "GradientBoosting": joblib.load("PeptideK_GradientBoosting.joblib"),
            "AdaBoost":         joblib.load("PeptideK_AdaBoost.joblib"),
            "SVM":              joblib.load("PeptideK_SVM.joblib"),
            "XGBoost":          joblib.load("PeptideK_XGBoost.joblib"),
        }
    }

models = load_models()

# ======================
# 2. Feature Extraction
# ======================
def calculate_charge(sequence):
    positive = sequence.count('K') + sequence.count('R') + sequence.count('H')
    negative = sequence.count('D') + sequence.count('E')
    return positive - negative

def calculate_hydrophobicity(sequence):
    hydrophobic_residues = 'AVILMFYW'
    return sum(sequence.count(aa) for aa in hydrophobic_residues)

def calculate_molecular_weight(sequence):
    return molecular_weight(Seq(sequence), seq_type='protein')

def calculate_number_of_cysteines(sequence):
    return sequence.count('C')

def calculate_number_of_disulfide_bridges(sequence):
    return calculate_number_of_cysteines(sequence) // 2

def calculate_isoelectric_point(sequence):
    pKa_acidic = {'D':3.9, 'E':4.25}
    pKa_basic = {'K':10.5, 'R':12.5, 'H':6.0}
    acidic_count = sum(sequence.count(aa) for aa in pKa_acidic)
    basic_count = sum(sequence.count(aa) for aa in pKa_basic)
    if acidic_count + basic_count == 0:
        return 7.0
    total_charge_pI = sum(pKa_acidic[aa]*sequence.count(aa) for aa in pKa_acidic) + \
                      sum(pKa_basic[aa]*sequence.count(aa) for aa in pKa_basic)
    return total_charge_pI / (acidic_count + basic_count)

def amino_acid_composition(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    total = len(sequence)
    return {aa: (sequence.count(aa) / total * 100 if total > 0 else 0) for aa in amino_acids}

def secondary_structure_features(sequence):
    helix_aa = 'ALIVMFYW'
    sheet_aa = 'FYW'
    turn_aa = 'GP'
    helix = sum(sequence.count(aa) for aa in helix_aa)
    sheet = sum(sequence.count(aa) for aa in sheet_aa)
    turn = sum(sequence.count(aa) for aa in turn_aa)
    flexibility = helix / len(sequence) if len(sequence) > 0 else 0
    return helix, turn, sheet, flexibility

X_train_cols = ['Length', 'Charge', 'Hydrophobicity', 'Molecular_Weight',
                'Number_of_Cysteines', 'Number_of_Disulfide_Bridges', 'Isoelectric_Point',
                'Helix', 'Turn', 'Sheet', 'Flexibility'] + list('ACDEFGHIKLMNPQRSTVWY')

def calc_features_dict(seq):
    features = {
        'Length': len(seq),
        'Charge': calculate_charge(seq),
        'Hydrophobicity': calculate_hydrophobicity(seq),
        'Molecular_Weight': calculate_molecular_weight(seq),
        'Number_of_Cysteines': calculate_number_of_cysteines(seq),
        'Number_of_Disulfide_Bridges': calculate_number_of_disulfide_bridges(seq),
        'Isoelectric_Point': calculate_isoelectric_point(seq),
    }
    features.update(amino_acid_composition(seq))
    helix, turn, sheet, flexibility = secondary_structure_features(seq)
    features.update({'Helix': helix, 'Turn': turn, 'Sheet': sheet, 'Flexibility': flexibility})
    return features

def prepare_features(seq):
    features = calc_features_dict(seq)
    return np.array([features.get(col, 0) for col in X_train_cols])

# ======================
# 3. Prediction Function
# ======================
def classify_peptide(sequence, dataset="PeptideP"):
    feature_vector = prepare_features(sequence).reshape(1, -1)

    model_probs = {}
    for model_name, model in models[dataset].items():
        try:
            prob = model.predict_proba(feature_vector)[:,1][0]
        except:
            prob = model.decision_function(feature_vector)
            prob = 1 / (1 + np.exp(-prob))  # sigmoid transform
            prob = prob[0]
        model_probs[model_name] = prob

    active = all(prob >= 0.5 for prob in model_probs.values())

    return {
        "Sequence": sequence,
        "Dataset": dataset,
        "Probabilities": model_probs,
        "Active": active
    }

# ======================
# 4. Streamlit UI
# ======================
st.title("🔬 Peptide Activity Classifier")

seq_input = st.text_input("Enter peptide sequence:", "")
dataset_choice = st.selectbox("Select dataset:", ["PeptideP", "PeptideE", "PeptideK"])

if st.button("Classify"):
    if not seq_input.strip():
        st.warning("⚠️ Please enter a sequence.")
    else:
        result = classify_peptide(seq_input.strip().upper(), dataset=dataset_choice)
        st.subheader("Results")
        st.write(f"**Dataset:** {result['Dataset']}")
        st.write(f"**Sequence:** {result['Sequence']}")
        st.write("### Probabilities:")
        for m, p in result["Probabilities"].items():
            st.write(f"- {m}: {p:.3f}")
        st.write(f"**Is Active?** {'✅ Yes' if result['Active'] else '❌ No'}")
