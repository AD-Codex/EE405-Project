
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Given points
# points = [(1, 2), (4, 2), (4, 3)]

def quard_regresion(points):
    x = np.array([point[0] for point in points]).reshape(-1, 1)
    y = np.array([point[1] for point in points])

    # Transform features to include quadratic term (x^2)
    poly_features = PolynomialFeatures(degree=1)
    x_poly = poly_features.fit_transform(x)

    # Fit quadratic regression model
    model = LinearRegression().fit(x_poly, y)

    # Generate new points along the quadratic regression curve
    x_new = np.linspace(min(x), max(x), num=10).reshape(-1, 1)
    x_new_poly = poly_features.transform(x_new)
    y_new = model.predict(x_new_poly)

    # Combine x_new and y_new into new_points
    new_points = list(zip(x_new.flatten(), y_new))

    rounded_points=[]

    print(new_points)
    for a in new_points:
        x_cor=round(a[0])
        y_cor=round(a[1])
        rounded_points.append((x_cor,y_cor))

    # print("New points on the regression line:")
    # print(rounded_points)
    return rounded_points


# print(quard_regresion(points))

# # Plot original points
# plt.scatter(x, y, color='blue', label='Original Points')

# # Plot quadratic regression curve
# plt.plot(x_new, y_new, color='red', label='Quadratic Regression')

# # Set labels and title
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Quadratic Regression')
# plt.legend()

# # Show plot
# plt.grid(True)
# plt.show()