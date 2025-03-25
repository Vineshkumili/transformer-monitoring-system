# transformer-monitoring-system
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from imblearn.over_sampling import SMOTE

# Load the dataset from a CSV file
df = pd.read_csv('/content/Health index2.csv')

# Function to categorize health index into conditions
def categorize_health_index(health_index):
    if 85 < health_index <= 100:
        return 'Very Good'
    elif 70 < health_index <= 85:
        return 'Good'
    elif 50 < health_index <= 70:
        return 'Fair'
    elif 30 < health_index <= 50:
        return 'Poor'
    elif 0 <= health_index <= 30:
        return 'Very Poor'
    else:
        return 'Unknown'

# Apply the categorization function to create a new column in the DataFrame
df["Condition"] = df["Health index"].apply(categorize_health_index)

# Prepare features (X) and target variable (y)
X = df.drop(columns=["Health index", "Condition"])  # Features: all columns except 'Health index' and 'Condition'
y = df["Condition"]  # Target: health condition

# Encode the target variable into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check the minimum class count to handle class imbalance
min_class_samples = min(pd.Series(y_encoded).value_counts())

# Apply SMOTE to balance the classes in the dataset
# Set k_neighbors to a safe value based on the minimum class count
k_neighbors_value = min(4, min_class_samples - 1) if min_class_samples > 1 else 1
smote = SMOTE(random_state=42, k_neighbors=k_neighbors_value)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a Random Forest classifier on the training data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model and label encoder to disk for future use
joblib.dump(model, "health_condition_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Function to predict health condition based on user input
def predict_condition_from_input():
    # Load the trained model and label encoder
    model = joblib.load("health_condition_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Define the feature names expected from user input
    feature_names = [
        "Hydrogen", "Oxygen", "Nitrogen", "Methane", "CO", "CO2", 
        "Ethylene", "Ethane", "Acetylene", "DBDS", "Power factor", 
        "Interfacial V", "Dielectric rigidity", "Water content"
    ]

    print("Please enter the following values for prediction:")
    user_input = []
    
    # Collect user input for each feature
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"{feature}: "))  # Prompt user for input
                user_input.append(value)  # Add the input value to the list
                break  # Exit the loop if input is valid
            except ValueError:
                print("Invalid input. Please enter a numeric value.")  # Handle invalid input

    # Make a prediction based on the user input
    prediction = model.predict([user_input])
    condition = label_encoder.inverse_transform(prediction)[0]  # Decode the predicted condition
    print(f"Predicted Health Condition: {condition}")  # Display the prediction

# Uncomment the line below to enable user input prediction
# predict_condition_from_input()

import joblib

# Load the trained model and label encoder from disk
model = joblib.load("health_condition_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define the feature values for a sample input
# These values correspond to the features used in the model
# Example values for a "Very Good" health condition
user_input = [
    13500,  # Hydrogen
    343,    # Oxygen
    36500,  # Nitrogen
    3150,   # Methane
    113,    # CO
    984,    # CO2
    5,      # Ethylene
    1230,   # Ethane
    1,      # Acetylene
    1.0,    # DBDS
    4.93,   # Power factor
    37,     # Interfacial V
    52,     # Dielectric rigidity
    6       # Water content
]

# Make a prediction using the trained model
prediction = model.predict([user_input])

# Decode the predicted condition back to the original label
condition = label_encoder.inverse_transform(prediction)[0]

# Print the predicted health condition to the user
print(f"Predicted Health Condition: {condition}")
