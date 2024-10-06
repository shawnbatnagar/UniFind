import pandas as pd
import numpy as np
from math import ceil
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Read in data frame, remove NaN values
df = pd.read_csv("/Users/aniru/Documents/UniFind/Colleges.csv", skiprows=1)
df = df.dropna()

# Selecting the columns that we need
selected_columns = ['institution name',
                    'ADM2022.SAT Evidence-Based Reading and Writing 50th percentile score',
                    'ADM2022.SAT Math 50th percentile score',
                    'ADM2022.ACT English 50th percentile score',
                    'ADM2022.ACT Math 50th percentile score',
                    'SFA2122.Average net price (income 0-30,000)-students awarded Title IV federal financial aid, 2021-22',
                    'SFA2122.Average net price (income 30,001-48,000)-students awarded Title IV federal financial aid, 2021-22',
                    'SFA2122.Average net price (income 48,001-75,000)-students awarded Title IV federal financial aid, 2021-22',
                    'SFA2122.Average net price (income 75,001-110,000)-students awarded Title IV federal financial aid, 2021-22',
                    'SFA2122.Average net price (income over 110,000)-students awarded Title IV federal financial aid, 2021-22']
df = df[selected_columns]

# Renaming score columns
df = df.rename(columns={'ADM2022.SAT Evidence-Based Reading and Writing 50th percentile score': 'SAT Reading',
                        'ADM2022.SAT Math 50th percentile score': 'SAT Math',
                        'ADM2022.ACT Composite 50th percentile score': 'ACT Composite',
                        'ADM2022.ACT English 50th percentile score': 'ACT Reading',
                        'ADM2022.ACT Math 50th percentile score': 'ACT Math',
                        'SFA2122.Average net price (income 0-30,000)-students awarded Title IV federal financial aid, 2021-22': '0-30,000 Net price',
                        'SFA2122.Average net price (income 30,001-48,000)-students awarded Title IV federal financial aid, 2021-22': '30,001-48,000 Net price',
                        'SFA2122.Average net price (income 48,001-75,000)-students awarded Title IV federal financial aid, 2021-22': '48,000-75,000 Net price',
                        'SFA2122.Average net price (income 75,001-110,000)-students awarded Title IV federal financial aid, 2021-22': '75,001-110,000 Net price',
                        'SFA2122.Average net price (income over 110,000)-students awarded Title IV federal financial aid, 2021-22': '110,000+ Net price'})

df1 = df
df = df.drop(columns='institution name')

st.title("UniFind")
st.title("Affordable universities tailored for you!")
st.markdown("---")

st.subheader("Your Preferences")
# Input section
stanTest = st.radio("Test Type", options=["SAT", "ACT"])

if stanTest == "SAT":
    english = st.number_input("SAT Evidence-Based Reading and Writing", value=500, step=10)
    math = st.number_input("SAT Math", value=500, step=10)
    df = df.drop(columns=['ACT Reading', 'ACT Math'])

else:
    english = st.number_input("ACT English", value=20, step=1)
    math = st.number_input("ACT Math", value=20, step=1)
    df = df.drop(columns=['SAT Reading', 'SAT Math'])

income = st.number_input("Annual Income", value=50000, step=1000)
net_price = st.number_input("Desired Net Price", value=15000, step=500)

# Selecting the income threshold column based on the users income
if income <= 30000:
    df = df.drop(columns=['30,001-48,000 Net price', '48,000-75,000 Net price', '75,001-110,000 Net price',
                          '110,000+ Net price'])
    desiredincome = "0-30,000 Net price"

elif income <= 48000:
    df = df.drop(columns=['0-30,000 Net price', '48,000-75,000 Net price', '75,001-110,000 Net price',
                          '110,000+ Net price'])
    desiredincome = "30,001-48,000 Net price"

elif income <= 75000:
    df = df.drop(columns=['0-30,000 Net price', '30,001-48,000 Net price', '75,001-110,000 Net price',
                          '110,000+ Net price'])
    desiredincome = "48,000-75,000 Net price"

elif income <= 110000:
    df = df.drop(columns=['0-30,000 Net price', '30,001-48,000 Net price', '48,000-75,000 Net price',
                          '110,000+ Net price'])
    desiredincome = "75,001-110,000 Net price"

else:
    df = df.drop(columns=['0-30,000 Net price', '30,001-48,000 Net price', '48,000-75,000 Net price',
                          '75,001-110,000 Net price'])
    df = df.rename("110,000+ Net price", "Net Price")
    desiredincome = "110,000+ Net price"

# User Data
user_data = {f"{stanTest} Reading": english, f"{stanTest} Math": math, desiredincome: net_price}
# Converting user data from a dictionary into a pandas dataframe
new_df = pd.DataFrame(user_data, [0])

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(df)

# Initialize the KNN model
k = 5  # Number of similar colleges to recommend
knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')

# Fit the model to your scaled college data
knn_model.fit(X_train_scaled)

# Scale the user inputs using the same scaler
user_inputs_scaled = scaler.transform(new_df)

# Find the k nearest neighbors to the scaled user inputs
distances, indices = knn_model.kneighbors(user_inputs_scaled)

# Computing similarity - distance is the dissimilarity, so take the opposite of that
def compute_similarity(distance):
    return 1 - distance

similarities = [compute_similarity(d) for d in distances]
if (sum(similarities[0]) / len(similarities[0])) * 100 < 0:
    output = 0
else:
    output = (sum(similarities[0]) / len(similarities[0])) * 100

strOutput = (str(output))[0:5]

# Get the indices of the recommended colleges
recommended_colleges_indices = indices[0]

recommended_colleges = df1.iloc[recommended_colleges_indices]
st.markdown("---")
st.subheader("Recommended Colleges")
st.write("We found universities that match your preferences by ", strOutput + "%")
recommended_colleges = recommended_colleges.rename(columns={'institution name': 'Institution Name'})
# Print recommended colleges, selecting income threshold as per user input
if income <= 30000:
    recommended_colleges = recommended_colleges.rename(columns={'0-30,000 Net price': 'Net Price'})

elif income <= 48000:
    recommended_colleges = recommended_colleges.rename(columns={'30,001-48,000 Net price': 'Net Price'})

elif income <= 75000:
    recommended_colleges = recommended_colleges.rename(columns={'48,000-75,000 Net price': 'Net Price'})

elif income <= 110000:
    recommended_colleges = recommended_colleges.rename(columns={'75,001-110,000 Net price': 'Net Price'})

else:
    recommended_colleges = recommended_colleges.rename(columns={'110,000+ Net price': 'Net Price'})
if stanTest=="SAT":
    testColumn1 = "SAT Reading"
    testColumn2 = "SAT Math"
else:
    testColumn1 = "ACT Reading"
    testColumn2 = "ACT Math"
st.write(recommended_colleges[['Institution Name','Net Price', testColumn1, testColumn2]])
# Print recommended
