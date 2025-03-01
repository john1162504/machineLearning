import numpy as np

n = [2,3,3,1]

W1 = np.random.randn(n[1],n[0])
W2 = np.random.randn(n[2],n[1])
W3 = np.random.randn(n[3],n[2])

b1 = np.random.randn(n[1],1)
b2 = np.random.randn(n[2],1)
b3 = np.random.randn(n[3],1)

X = np.array([
    [150, 70],
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]
])

A0 = X.T

y = np.array([
    0,  # whew, thank God Jimmy isn't at risk for cardiovascular disease.
    1,   # damn, this guy wasn't as lucky
    1, # ok, this guy should have seen it coming. 5"8, 312 lbs isn't great.
    0,
    0,
    1,
    1,
    0,
    1,
    0
])
m = 10

# we need to reshape to a n^[3] x m matrix
Y = y.reshape(n[3], m)

def sigmoid(arr):
  return 1 / (1 + np.exp(-1 * arr))

# layer 1 calculations

Z1 = W1 @ A0 + b1  # the @ means matrix multiplication

assert Z1.shape == (n[1], m) # just checking if shapes are good
A1 = sigmoid(Z1)

# layer 2 calculations
Z2 = W2 @ A1 + b2
assert Z2.shape == (n[2], m)
A2 = sigmoid(Z2)

# layer 3 calculations
Z3 = W3 @ A2 + b3
assert Z3.shape == (n[3], m)
A3 = sigmoid(Z3)


y_hat = A3 

def cost(y_hat, y):
  """
  y_hat should be a n^L x m matrix
  y should be a n^L x m matrix
  """
  # 1. losses is a n^L x m
  losses = - ( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )

  m = y_hat.reshape(-1).shape[0]

  # 2. summing across axis = 1 means we sum across rows, 
  #   making this a n^L x 1 matrix
  summed_losses = (1 / m) * np.sum(losses, axis=1)

  # 3. unnecessary, but useful if working with more than one node
  #   in output layer
  return np.sum(summed_losses)
