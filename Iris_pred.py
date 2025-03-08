import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: species

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict species based on user input
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                               columns=iris.feature_names)
    predicted_species = model.predict(input_data)
    species_name = iris.target_names[predicted_species][0]
    return species_name

# Interactive input loop
while True:
    try:
        sepal_length = float(input("Enter sepal length (cm): "))
        sepal_width = float(input("Enter sepal width (cm): "))
        petal_length = float(input("Enter petal length (cm): "))
        petal_width = float(input("Enter petal width (cm): "))
        
        species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
        print(f'The predicted species is: {species}')
        
        continue_input = input("Do you want to enter another flower? (yes/no): ").strip().lower()
        if continue_input != 'yes':
            break
    except ValueError:
        print("Invalid input. Please enter numeric values.")
