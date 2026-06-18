import os
import threading
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["POST", "GET", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

model = None
model_columns = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATHS = [
    os.path.join(BASE_DIR, 'BankChurners.csv'),
    os.path.join(BASE_DIR, 'credit_card_churn.csv'),
    os.path.join(BASE_DIR, 'dataset.csv')
]

DEFAULT_VALUES = {
    'Customer_Age': 40,
    'Gender': 'M',
    'Education_Level': 'Graduate',
    'Marital_Status': 'Married',
    'Income_Category': '$80K - $120K',
    'Card_Category': 'Blue',
    'Credit_Limit': 5000,
    'Avg_Open_To_Buy': 2500,
    'Total_Trans_Ct': 60,
    'Total_Revolving_Bal': 1500,
    'Months_Inactive_12_mon': 2,
    'Contacts_Count_12_mon': 2,
    'Total_Amt_Chng_Q4_Q1': 1.15
}


def find_dataset_path():
    for path in DATA_PATHS:
        if os.path.exists(path):
            return path
    return None


def preprocess_dataframe(df):
    df = df.copy()
    df = df.drop(columns=[
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ], errors='ignore')

    if 'CLIENTNUM' in df.columns:
        df = df.drop(columns=['CLIENTNUM'])

    df = df[df != 'Unknown'].dropna()
    return df


def train_model():
    global model, model_columns
    dataset_path = find_dataset_path()

    if dataset_path is None:
        app.logger.warning('Nenhum dataset encontrado. O backend ficará disponível apenas como placeholder.')
        return

    df = pd.read_csv(dataset_path)
    df = preprocess_dataframe(df)

    if 'Attrition_Flag' not in df.columns:
        raise RuntimeError('O dataset precisa conter a coluna Attrition_Flag.')

    y = (df['Attrition_Flag'] == 'Attrited Customer').astype(int)
    X = df.drop(columns=['Attrition_Flag'], errors='ignore').copy()
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scale_pos_weight_val = sum(y_train == 0) / sum(y_train == 1)
    n_estimators_values = [500, 750, 1000, 1250, 1500]
    results = []

    app.logger.info('Buscando melhor n_estimators com validação cruzada (CV=5) no treino...')
    for n_est in n_estimators_values:
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=n_est,
            random_state=42,
            scale_pos_weight=scale_pos_weight_val
        )

        cv_scores = cross_val_score(
            xgb_model, X_train, y_train,
            cv=5, scoring='f1'
        )

        results.append({
            'n_estimators': n_est,
            'f1_medio_cv': cv_scores.mean(),
            'f1_std_cv': cv_scores.std()
        })

    results_df = pd.DataFrame(results)
    best_n = int(results_df.loc[results_df['f1_medio_cv'].idxmax(), 'n_estimators'])
    app.logger.info(f'Melhor n_estimators encontrado: {best_n}.')

    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=best_n,
        random_state=42,
        scale_pos_weight=scale_pos_weight_val
    )
    final_model.fit(X_train, y_train)

    model_columns = X.columns.tolist()
    model = final_model
    app.logger.info(f'Modelo final treinado com {len(model_columns)} features.')


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    global model, model_columns

    if request.method == 'OPTIONS':
        return '', 200

    if model is None:
        return jsonify({
            'error': 'Modelo não está disponível. Coloque o arquivo CSV do dataset em data/credit_card_churn.csv ou na raiz.'
        }), 500

    data = request.get_json() or {}
    sample = {}

    for key, default in DEFAULT_VALUES.items():
        value = data.get(key, default)
        sample[key] = value

    sample_df = pd.DataFrame([sample])
    sample_df = preprocess_dataframe(sample_df)
    sample_df = pd.get_dummies(sample_df, drop_first=True)

    for col in model_columns:
        if col not in sample_df.columns:
            sample_df[col] = 0
    sample_df = sample_df[model_columns]

    probability = float(model.predict_proba(sample_df)[0, 1])
    label = 'Attrited Customer' if probability >= 0.5 else 'Existing Customer'
    confidence = f'{probability:.2f}'

    return jsonify({
        'probability': probability,
        'label': label,
        'confidence': confidence,
        'input': sample
    })


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'Acesse /api/status para ver o estado do modelo.'
    })


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'model_loaded': model is not None,
        'message': 'API de previsão está pronta.'
    })


def run_training_in_background():
    try:
        print("Iniciando treinamento do modelo XGBoost em background...")
        train_model()
        print("Treinamento em background concluído com sucesso!")
    except Exception as e:
        print(f"Erro ao treinar o modelo em background: {e}")


training_thread = threading.Thread(target=run_training_in_background)
training_thread.daemon = True

if __name__ == '__main__':
    training_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
else:
    training_thread.start()