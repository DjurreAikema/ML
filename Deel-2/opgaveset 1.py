import numpy as np
from sklearn.datasets import load_iris

# Stap 1: Download de dataset m.b.v. load_iris() uit sklearn.datasets.
iris = load_iris()

# Stap 2: Vul je featurematrix X op basis van de data.
X = iris.data

# Stap 3: De uitkomstvector y ga je vullen op basis van target. Standaard bevat deze array de waardes 0, 1 en 2 (resp. 'setosa', 'versicolor', 'virginica').
# Maak deze binair door 0 en 1 allebei 0 te maken (niet-virginica) en van elke 2 een 1 te maken (wel-virginica). Denk erom dat y het juiste datatype en de juiste shape krijgt.
y = iris.target
y = np.where(y == 2, 1, 0) # Virginica wordt 1, de rest 0
y = y.reshape(-1, 1) # Zorg ervoor dat y de juiste vorm heeft


# Stap 4: Definieer een functie sigmoid() die de sigmoïde-functie implementeert.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Stap 5: Initialiseer een vector theta met 1.0'en in de juiste shape.
theta = np.ones((X.shape[1], 1))

# Stap 6: Nu kun je beginnen aan de loop waarin je in 1500 iteraties:
alpha = 0.01  # Learning rate
num_iterations = 1500  # Aantal iteraties

for _ in range(num_iterations):
    # Stap 6.1: De voorspellingen (denk aan sigmoid!) en de errors berekent.
    predictions = sigmoid(np.dot(X, theta)) # Bereken de voorspellingen
    errors = predictions - y # Bereken de fouten (verschil tussen voorspelling en werkelijke waarden)

    # Stap 6.2: De gradient berekent en theta aanpast. Werk in eerste instantie met een learning rate van 0.01.
    gradient = np.dot(X.T, errors) / len(X) # Bereken de gradient van de kostenfunctie
    theta -= alpha * gradient # Pas theta aan met de learning rate en de gradient

    # Stap 6.3: De kosten berekent.
    cost = np.sum(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions)) / len(X) # Bereken de kosten
    print(cost) # Print de kosten voor elke iteratie

# Stap 7: Als het goed is, zie je de kosten (vanaf een beginwaarde rond 8) steeds dalen kom je aan het einde rond 0,24 uit.
# Werk je met de niet-negatieve versie van de kostenfunctie, dan ga ja van ongeveer -8 naar -0,24.

# Stap 8: Experimenteer eens met andere waardes van de learning rate (1.0 < alpha < 0.0) en het aantal iteraties.
