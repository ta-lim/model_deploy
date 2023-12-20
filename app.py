from flask import Flask, request
import json
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model and encoders
df_tourism = pd.read_csv('./list/df_tourism.csv')
model = load_model('./model/tourism_recommendation_model.h5')
category_encoder = LabelEncoder()
city_encoder = LabelEncoder()
category_encoder.classes_ = np.load('./encoder/category_encoder_classes.npy', allow_pickle=True)
city_encoder.classes_ = np.load('./encoder/city_encoder_classes.npy', allow_pickle=True)
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit_transform(df_tourism['Description'])

def recommend_places(user_interests):
    # Filter the tourism DataFrame based on user interests
    df_user_interests = df_tourism[df_tourism['Place_Id'].isin(user_interests['Place_Id'])]
    print(df_user_interests)
    
    # Get the indices of the user's interests to avoid recommending the same places
    indices_to_exclude = df_user_interests.index

    # Prepare the rest of the places for prediction
    df_other_places = df_tourism.drop(indices_to_exclude)
    other_places_features = np.hstack((
        to_categorical(category_encoder.transform(df_other_places['Category'])),
        to_categorical(city_encoder.transform(df_other_places['City'])),
        tfidf_vectorizer.transform(df_other_places['Description']).toarray()
    ))

    # Predict the likelihood of the user being interested in these other places
    predictions = model.predict(other_places_features)
    
    # Get the top 15 recommendations
    top_indices = predictions.flatten().argsort()[-45:][::-3]
    print(top_indices)
    recommendations = df_other_places.iloc[top_indices]
    
    return recommendations[['Place_Id','Place_Name', 'Description', 'Category', 'City']]



@app.after_request
def add_header(response):
    response.headers['Content-Type'] = 'application/json'
    return response

# Define the API endpoint for prediction
@app.route('/preferences', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_choices = data['Place_Id']
        user_interests = pd.DataFrame({'Place_Id': user_choices})

        # Use the recommend_places function
        recommendations = recommend_places(user_interests)

        # Convert recommendations to JSON
        recommendations_dict = recommendations.to_dict(orient='records')
        return json.dumps({'success': True, 'recommendations': recommendations_dict}, sort_keys=False)

    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)
