import matplotlib.pyplot as plt

theta_1 = 2
theta_0 = 9
alpha = 0.01
iterations = 7000
threshold = 0.001
its = 0

data_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
data_y = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

m = len(data_x)

def compute_cost():
    total_cost = 0
    for _ in range(m):
        h = theta_1 * data_x[_] + theta_0
        total_cost += (h - data_y[_]) ** 2
    return total_cost / (2 * m)

plt.scatter(data_x, data_y, color='blue', label='Data points')

for _ in range(iterations):
    gradient_0 = 0
    gradient_1 = 0
    for _ in range(m):
        h = theta_1 * data_x[_] + theta_0
        gradient_0 += (h - data_y[_])
        gradient_1 += (h - data_y[_]) * data_x[_]

    theta_0 -= (alpha / m) * gradient_0
    theta_1 -= (alpha / m) * gradient_1
    its += 1

    cost = compute_cost()
    if cost < threshold:
        break

    y_pred = [theta_1 * x + theta_0 for x in data_x]
    plt.plot(data_x, y_pred, color='red', alpha=0.01)

print(f"Final Parameters: theta_0 = {theta_0}, theta_1 = {theta_1}", end="\n\n")
print(f"Final Equation: Y = {theta_0} + {theta_1} * X", end="\n\n")
print(f"Total Number of iterations: {its} iterations")

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Over Iterations')
plt.show()
