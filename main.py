import numpy as np
import matplotlib.pyplot as plt

#step 1: generate fake data (x, y)
#we'll create 100 random x value between 0 and 10
x = np.random.rand(100, 1) * 10

#define the real relationship: y = 2x + 3 + noise
y = 2 * x + 3 + np.random.randn(100, 1) * 2 #small noise makes it more realistic

#step 2: visualize data
plt.scatter(x, y, color='orange', label='Data points')
plt.title("Generated Data(y = 2x + 3 + noise)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
# plt.savefig('plot.png')
# print("Plot saved as plot.png")

#step 3: define model parameters (random initial guess)
w = np.random.randn(1)
b = np.random.randn(1)

#step 4: define the model (hypothesis)
def predict(x):
    return w * x + b

#make some initial predictions
y_pred = predict(x)

#step 5: visualize predictions vs actual
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', label='Initial prediction')
plt.title("Model Hypothesis (random guess)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()