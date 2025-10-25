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

# step 6: define loss function (Mean Squared Error)
def compute_loss(y, Y_pred):
    return np.mean((y - y_pred) ** 2)

#compute current loss
loss = compute_loss(y, y_pred)
# print(f"Initial Loss: {loss:.4f}")

#step 7: compute gradients (derivatives)
def compute_gradients(x, y, y_pred):
    dw = (-2 / len(x)) * np.sum(x * (y - y_pred))
    db = (-2 / len(x)) * np.sum(y - y_pred)
    return dw, db

#step 8: Train using gradient descent
learning_rate = 0.001
epochs = 10000

for epoch in range(epochs):
    #forward pass
    y_pred = predict(x)

    #compute loss
    loss = compute_loss(y, y_pred)

    #compute gradients
    dw, db = compute_gradients(x, y, y_pred)

    #update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    # print progress occasionally
    if epoch  % 100 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, w={w[0]:.4f}, b={b[0]:.4f}")

# step 9: final prediction after training
y_final = predict(x)

plt.scatter(x, y, color='blue', label="Actual Data")
plt.plot(x, y_final, color='red', label="Learned line")
plt.title("Model after training")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"\nFinal Parameters: w={w[0]:.4f}, b={b[0]:.4f}")
