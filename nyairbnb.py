import pandas as pd
import numpy as np  

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import time

# Neural Network Model
class RegressionModel(nn.Module):
    def __init__(self, input_size=37, hidden_size=64, output_size=1):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size*2)  # Input layer with 1 feature, hidden layer with 64 units
        self.layer2 = nn.Linear(hidden_size*2, hidden_size)  # Hidden layer with 32 units
        self.layer3 = nn.Linear(hidden_size, hidden_size)  # Hidden layer with 32 units
        self.layer4 = nn.Linear(hidden_size, hidden_size)  # Hidden layer with 32 units
        self.output = nn.Linear(hidden_size, 1)  # Output layer for regression

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # ReLU activation for layer 1
        x = torch.relu(self.layer2(x))  # ReLU activation for layer 2
        x = torch.relu(self.layer3(x))  # ReLU activation for layer 2
        x = torch.relu(self.layer4(x))  # ReLU activation for layer 2
        x = self.output(x)  # Output layer
        return x
    


def get_data():
    random_state = 5000
    df2 = pd.read_csv('final_df.csv')
    df2['sentiment'] = df2['sentiment'].fillna(3)
    ###


    X = df2.drop(['actual_price','id', 'listing_id', 'Unnamed: 0'], axis=1)
    X = X.astype({col: 'float32' for col in X.select_dtypes(include='bool').columns})

    X['bedrooms'] = X['bedrooms'].replace(0, 1)
    X['beds'] = X['beds'].replace(0, 1)

    y = df2['actual_price']/X['accommodates']#normalize price by number of guests

    y = y/X['bedrooms']
    y = y/X['beds']

    X = X.drop(['accommodates', 'bedrooms','beds'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=random_state, shuffle=True)
    return X_train, X_test, y_train, y_test
'''
listings = pd.read_csv('listings.csv')
reviews = pd.read_csv('reviews.csv')
calendar = pd.read_csv('calendar.csv')
calendar.price = calendar.price.replace('[\$,]', '', regex=True).astype(float)
listings.price = listings.price.replace('[\$,]', '', regex=True).astype(float)
df1 = calendar.drop_duplicates(subset=['listing_id'], keep='first').set_index('listing_id')
df2 = listings.set_index('id')
df2['actual_price'] = df1['price']
df2.reset_index(inplace=True)
df2=df2[df2['actual_price']<500]
df2 = pd.get_dummies(df2, columns=['host_response_time',  'host_is_superhost', 'room_type', 'neighbourhood_cleansed'], drop_first=True)
droplist = ['listing_url', 'scrape_id', 'last_scraped', 'source', 'name',
       'description', 'neighborhood_overview', 'picture_url', 'host_id',
       'host_url', 'host_name', 'host_since', 'host_location', 'host_about',
        'host_acceptance_rate', 'host_thumbnail_url',
       'host_picture_url', 'host_neighbourhood', 'host_listings_count',
       'host_total_listings_count', 'host_verifications',
       'host_has_profile_pic', 'host_identity_verified', 'neighbourhood',
       'neighbourhood_group_cleansed', 
       'property_type',  'bathrooms_text',
        'amenities', 'price', 'minimum_nights',
       'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights',
       'minimum_maximum_nights', 'maximum_maximum_nights',
       'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated',
       'has_availability', 'availability_30', 'availability_60',
       'availability_90',  'calendar_last_scraped',
        'number_of_reviews_ltm', 'number_of_reviews_l30d',
       'first_review', 'last_review', 
        'license',
       'instant_bookable', 'calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms']

df2.host_response_rate = df2.host_response_rate.replace('[%]', '', regex=True).astype(float)
df2 =df2.drop(droplist, axis=1)
df2.fillna(0, inplace=True)


###
df2 = pd.read_csv('final_df.csv')
df2['sentiment'].fillna(3, inplace=True)
###


X = df2.drop(['actual_price','id', 'listing_id', 'Unnamed: 0'], axis=1)
X = X.astype({col: 'float32' for col in X.select_dtypes(include='bool').columns})
X['bedrooms'] = X['bedrooms'].replace(0, 1)
X['beds'] = X['beds'].replace(0, 1)

y = df2['actual_price']/X['accommodates']#normalize price by number of guests

y = y/X['bedrooms']
y = y/X['beds']

X = X.drop(['accommodates', 'bedrooms','beds'], axis=1)
y = df2['actual_price']
random_state = 5000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, shuffle=True, random_state=random_state)
'''
torch.manual_seed(1)
np.random.seed(1)
X_train, X_test, y_train, y_test = get_data()
# lasso regression
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.15, max_iter=1000)
lasso.fit(X_train, y_train)
for i in range (100):
    start_time = time.time()
    y_pred = lasso.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_score = lasso.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
end_time = time.time()
print(f'Time: {(end_time - start_time)/100:.6f} seconds')
print('RMSE: ', rmse)
print('R-Squared: ', r_score)
print('MAE: ', mae)
print('MAPE: ', mape)
lm = LinearRegression()

lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('RMSE: ', rmse)
# Convert to PyTorch tensors
#X = X.astype({col: 'float32' for col in X.select_dtypes(include='bool').columns})
#X_tensor = torch.tensor(X.values, dtype=torch.float32)
#y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, shuffle=True, random_state= random_state)
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)
# Initialize model
model = RegressionModel(X_train.shape[1], 8)
print(model)

# Loss and Optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# Training loop
epochs = 150
for epoch in range(epochs):
    model.train()

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(X_train)

    # Calculate loss
    loss = criterion(y_pred.squeeze(), y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()  # Switch model to evaluation mode
#record current time
start_time = time.time()
for i in range (100):
    with torch.no_grad():
        y_pred = model(X_test)

    # Convert predictions and true values to numpy for evaluation
    y_pred_np = y_pred.numpy()
    y_test_np = y_test.numpy()

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mse)
    r_score = r2_score(y_test_np, y_pred_np)
    mae = mean_absolute_error(y_test_np, y_pred_np)
    mape = mean_absolute_percentage_error(y_test_np, y_pred_np)
end_time = time.time()
print(f'Time: {(end_time - start_time)/100:.6f} seconds')

print(f'RootMean Squared Error: {rmse:.4f}, R-Score: {r_score:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}') 
'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Using SST-2 model (binary, modify with thresholds)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
revcnt = 0
def sentiment_analysis(review):
    global revcnt
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = softmax(outputs.logits.detach().numpy())[0]  # Apply softmax to get probabilities
    sentiment = 3
    # Define thresholds and classify into 5 classes based on scores
    if scores[1] > 0.9:  # high confidence for positive
        sentiment = 5
    elif scores[1] > 0.6:
        sentiment = 4
    elif scores[0] > 0.9:  # high confidence for negative
        sentiment = 1
    elif scores[0] > 0.6:
        sentiment = 3
    revcnt += 1
    if revcnt % 1000 == 0:
        print(f"Sentiment: {sentiment}, review count: {revcnt}")  # Print the sentiment)
    return sentiment

#reviews = reviews[reviews['listing_id'].isin(df2['id'])]
#reviews.dropna(subset=['comments'], inplace=True)
#reviews['sentiment'] = reviews['comments'].apply(sentiment_analysis)
#mean_sentiment = reviews.groupby('listing_id')['sentiment'].mean().reset_index()
#df2 = df2.merge(mean_sentiment, left_on='id', right_on='listing_id',how='left')
#df2.to_csv('df2.csv')
print(df2.shape)
print(df2[['sentiment']].describe())
'''