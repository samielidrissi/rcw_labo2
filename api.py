from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

def load_data(data, delimiter=',', has_header='yes'):
    try:
        if delimiter == ',':
            if has_header == 'yes':
                dataframe = pd.read_csv(data)
            else:
                dataframe = pd.read_csv(data, header=None)
                columns = [f"Column {i + 1}" for i in range(dataframe.shape[1])]
                dataframe.columns = columns
        elif delimiter == ' ':
            if has_header == 'yes':
                dataframe = pd.read_csv(data, delim_whitespace=True)
            else:
                dataframe = pd.read_csv(data, delim_whitespace=True, header=None)
                columns = [f"Column {i + 1}" for i in range(dataframe.shape[1])]
                dataframe.columns = columns
        return dataframe
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return str(e)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    delimiter = request.form.get('delimiter', ',')
    has_header = request.form.get('has_header', 'yes')
    
    data = load_data(file, delimiter, has_header)
    if isinstance(data, str):
        logging.error(f"Error in data: {data}")
        return jsonify({'error': data})
    
    data.to_pickle('data.pkl')
    logging.info("File uploaded successfully")
    return jsonify({'message': 'File uploaded successfully'})

@app.route('/train', methods=['POST'])
def train_model():
    try:
        model_type = request.json.get('model_type')
        model_name = request.json.get('model_name')
        data = pd.read_pickle('data.pkl')
        
        X = data.iloc[:, :-1]
        Y = data.iloc[:, -1]
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=7)
        
        if model_type == 'Classification':
            models = {
                'LR': LogisticRegression(solver='newton-cg'),
                'LDA': LinearDiscriminantAnalysis(),
                'Decision Tree': DecisionTreeClassifier(),
                'KNN': KNeighborsClassifier(n_neighbors=11),
                'SVM': SVC(),
                'Random Forest': RandomForestClassifier(),
                'AdaBoost': AdaBoostClassifier()
            }
        elif model_type == 'Regression':
            models = {
                'LR': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'Elastic': ElasticNet(),
                'KNN': KNeighborsRegressor(),
                'Tree': DecisionTreeRegressor(),
                'SVR': SVR()
            }
        else:
            logging.error("Unsupported model type")
            return jsonify({'error': 'Unsupported model type'})
        
        if model_name not in models:
            logging.error("Unsupported model name")
            return jsonify({'error': 'Unsupported model name'})
        
        model = models[model_name]
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        
        if model_type == 'Classification':
            matrix = confusion_matrix(Y_test, predictions)
            TN, FN, FP, TP = matrix.ravel()
            accuracy = (TP + TN) / (TN + FN + FP + TP)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            specificity = TN / (TN + FP)
            f1_score = 2 * ((precision * recall) / (precision + recall))
            metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'F1-Score': f1_score
            }
        elif model_type == 'Regression':
            mae = mean_absolute_error(Y_test, predictions)
            mse = mean_squared_error(Y_test, predictions)
            r2 = r2_score(Y_test, predictions)
            metrics = {
                'MAE': mae,
                'MSE': mse,
                'R2 Score': r2
            }
        
        with open('final_model.pickle', 'wb') as file:
            pickle.dump(model, file)
        
        logging.info("Model trained and saved successfully")
        return jsonify(metrics)
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
