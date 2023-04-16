import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# define the objective function to be minimized
def objective_function(X, y, features):
    # create a LogisticRegression object with fit_intercept=False
    logreg = LogisticRegression(fit_intercept=False)

    # select only the features that are currently turned on
    selected_features = X[:, features]

    # train the model on the selected features
    logreg.fit(selected_features, y)

    # calculate the accuracy of the model
    accuracy = logreg.score(selected_features, y)

    # return the negative of the accuracy (since we want to minimize the objective function)
    return -accuracy

# set the parameters for simulated annealing
n_features = 20
n_iterations = 1000
initial_temperature = 10.0
temperature_decay = 0.95

# generate some random data
X, y = make_classification(n_samples=100, n_features=n_features, n_informative=10, n_classes=2, random_state=42)

# initialize the current state to use all features
current_features = np.ones(n_features, dtype=bool)

# initialize the current score
current_score = objective_function(X, y, current_features)

# initialize the best score and best features to the current score and features
best_score = current_score
best_features = current_features.copy()

# perform simulated annealing
for i in range(n_iterations):
    # select a random feature to toggle
    idx = np.random.randint(0, n_features)

    # create a copy of the current state and toggle the selected feature
    next_features = current_features.copy()
    next_features[idx] = not next_features[idx]

    # calculate the score for the new state
    next_score = objective_function(X, y, next_features)

    # calculate the acceptance probability
    delta_score = next_score - current_score
    temperature = initial_temperature * (temperature_decay ** i)
    acceptance_probability = np.exp(np.clip(delta_score / temperature, -100, 100))

    # decide whether to move to the new state or stay in the current state
    if delta_score > 0 or np.random.rand() < acceptance_probability:
        current_features = next_features
        current_score = next_score

    # update the best score and best features if the current score is better
    if current_score > best_score:
        best_score = current_score
        best_features = current_features.copy()

# create a LogisticRegression object with fit_intercept=False and the best features
logreg = LogisticRegression(fit_intercept=False)
selected_features = X[:, best_features]
logreg.fit(selected_features, y)

# print the accuracy of the model
accuracy = logreg.score(selected_features, y)
print(f"Best features: {best_features}")
print(f"Accuracy: {accuracy}")

# create a heatmap of the scores for each feature subset
import matplotlib.pyplot as plt
score_matrix = np.zeros((n_iterations, n_features))
for i in range(n_iterations):
    for j in range(n_features):
        features = np.ones(n_features, dtype=bool)
        features[j] = not features[j]
        score_matrix[i, j] = objective_function(X, y, features)
plt.imshow(score_matrix, cmap='coolwarm', aspect='auto')
plt.xlabel('Features')
plt.ylabel('Iterations')
plt.colorbar()
plt.show()
