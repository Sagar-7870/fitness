import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load and preprocess data
def load_data():
    calories_path = 'calories.csv'
    exercise_path = 'exercise.csv'
    calories_data = pd.read_csv("C:/Users/ACER/Documents/Fitness tracker/fitness/calories.csv")
    exercise_data = pd.read_csv("C:/Users/ACER/Documents/Fitness tracker/fitness/exercise.csv")
    
    # Merge datasets on User_ID
    data = pd.merge(exercise_data, calories_data, on='User_ID')
    return data

# Display data summary
def display_summary(data):
    st.write("### Fitness Tracker Summary")
    st.write(f"**Total Users:** {data['User_ID'].nunique()}")
    st.write(f"**Average Calories Burned:** {data['Calories'].mean():.2f}")
    st.write(f"**Average Exercise Duration:** {data['Duration'].mean():.2f} minutes")
    st.write(f"**Average Heart Rate:** {data['Heart_Rate'].mean():.2f} bpm")
    st.write("**Gender Distribution:**")
    st.write(data['Gender'].value_counts())

# Plot insights
def plot_insights(data):
    # Plot calories burned by age group
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data, x='Age', bins=10, kde=True, hue='Gender', ax=ax)
    ax.set_title('Age Distribution by Gender')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Scatter plot: Exercise Duration vs Calories
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='Duration', y='Calories', hue='Gender', ax=ax)
    ax.set_title('Exercise Duration vs. Calories Burned')
    ax.set_xlabel('Duration (minutes)')
    ax.set_ylabel('Calories Burned')
    st.pyplot(fig)

# Train a machine learning model
def train_model(data):
    # Select features and target
    features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    target = 'Calories'
    
    X = data[features]
    y = data[target]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    st.write(f"### Model Trained Successfully")
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    
    return model

# Predict calories based on user input
def predict_calories(model):
    st.write("### Predict Calories Burned")
    age = st.number_input("Enter age:", min_value=0, value=25, step=1)
    height = st.number_input("Enter height (cm):", min_value=50, value=170, step=1)
    weight = st.number_input("Enter weight (kg):", min_value=10, value=70, step=1)
    duration = st.number_input("Enter exercise duration (minutes):", min_value=0, value=30, step=1)
    heart_rate = st.number_input("Enter average heart rate:", min_value=0, value=120, step=1)
    body_temp = st.number_input("Enter body temperature:", min_value=0.0, value=36.5, step=0.1)
    
    if st.button("Predict Calories"):
        # Create input array
        input_data = np.array([[age, height, weight, duration, heart_rate, body_temp]])
        prediction = model.predict(input_data)[0]
        st.write(f"**Predicted Calories Burned:** {prediction:.2f} kcal")

# Main function
def main():
    st.title("Enhanced Fitness Tracker App")
    data = load_data()

    menu = ["Summary", "Insights", "Train Model", "Predict Calories"]
    choice = st.sidebar.selectbox("Select an Option", menu)

    if choice == "Summary":
        display_summary(data)
    elif choice == "Insights":
        plot_insights(data)
    elif choice == "Train Model":
        global model
        model = train_model(data)
    elif choice == "Predict Calories":
        if 'model' in globals():
            predict_calories(model)
        else:
            st.write("Please train the model first in the Train Model section.")

if __name__ == "__main__":
    main()