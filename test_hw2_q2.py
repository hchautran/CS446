import torch
import hw2_utils as utils
import hw2_q2

# Load training data
print("Loading training data...")
X_train, y_train = utils.gaussian_dataset("train")
print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

# Load test data
print("\nLoading test data...")
X_test, y_test = utils.gaussian_dataset("test")
print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

# Train the model (estimate parameters)
print("\nEstimating parameters...")
mu, sigma2 = hw2_q2.gaussian_theta(X_train, y_train)
p = hw2_q2.gaussian_p(y_train)

print(f"\nEstimated parameters:")
print(f"mu (class means):\n{mu}")
print(f"sigma2 (class variances):\n{sigma2}")
print(f"p (P(Y=0)): {p}")

# Classify training data
print("\nClassifying training data...")
y_train_pred = hw2_q2.gaussian_classify(mu, sigma2, p, X_train)
train_accuracy = (y_train_pred == y_train).float().mean()
print(f"Training accuracy: {train_accuracy:.4f} ({(y_train_pred == y_train).sum()}/{len(y_train)})")

# Classify test data
print("\nClassifying test data...")
y_test_pred = hw2_q2.gaussian_classify(mu, sigma2, p, X_test)
test_accuracy = (y_test_pred == y_test).float().mean()
print(f"Test accuracy: {test_accuracy:.4f} ({(y_test_pred == y_test).sum()}/{len(y_test)})")

# Confusion matrix for test set
print("\nTest set confusion matrix:")
TP = ((y_test_pred == 1) & (y_test == 1)).sum()
TN = ((y_test_pred == 0) & (y_test == 0)).sum()
FP = ((y_test_pred == 1) & (y_test == 0)).sum()
FN = ((y_test_pred == 0) & (y_test == 1)).sum()
print(f"True Positives: {TP}")
print(f"True Negatives: {TN}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
