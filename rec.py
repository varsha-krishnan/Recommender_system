from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset into a DataFrame
data = pd.read_csv('netflix_titles.csv')

# Create user and item index mapping
user_index = {user_id: index for index, user_id in enumerate(data['show_id'].unique())}
item_index = {item_id: index for index, item_id in enumerate(data['title'].unique())}

# Convert user and item IDs to indices
data['user_index'] = data['show_id'].map(user_index)
data['item_index'] = data['title'].map(item_index)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

# Define model inputs
num_users = len(user_index)
num_items = len(item_index)
embedding_dim = 10

user_input = tf.keras.Input(shape=(1,))
item_input = tf.keras.Input(shape=(1,))

# User and item embedding layers
user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)(user_input)
item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)(item_input)

# Flatten the embeddings
user_flatten = tf.keras.layers.Flatten()(user_embedding)
item_flatten = tf.keras.layers.Flatten()(item_embedding)

# Dot product of user and item embeddings
dot_product = tf.keras.layers.Dot(axes=1)([user_flatten, item_flatten])

# Create the model
model = tf.keras.Model(inputs=[user_input, item_input], outputs=dot_product)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Prepare training data
user_indices_train = train_data['user_index'].values
item_indices_train = train_data['item_index'].values
ratings_train = np.ones(len(train_data))

# Fit the model
model.fit(x=[user_indices_train, item_indices_train], y=ratings_train, epochs=10, batch_size=1)

# Prepare testing data
user_indices_test = test_data['user_index'].values
item_indices_test = test_data['item_index'].values
ratings_test = np.ones(len(test_data))

# Evaluate the model
loss = model.evaluate(x=[user_indices_test, item_indices_test], y=ratings_test)
print('Test loss:', loss)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form.get('user_id')

    if user_id:
        user_idx = user_index[user_id]
        item_indices = np.arange(num_items)
        item_indices = np.delete(item_indices, user_idx)

        user_indices = np.full(len(item_indices), user_idx)

        predicted_ratings = model.predict([user_indices, item_indices])

        # Sort the predictions by predicted rating
        top_n = np.argsort(predicted_ratings.flatten())[::-1][:10]

        # Get the recommended item titles
        recommended_item_titles = [list(item_index.keys())[list(item_index.values()).index(item_idx)] for item_idx in item_indices[top_n]]

        return render_template('rec.html', user_id=user_id, recommendations=recommended_item_titles)
    else:
        return render_template('index.html', error_message='Please provide a user ID.')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
