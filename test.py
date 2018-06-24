import numpy as np
import fit_circle

import matplotlib.pyplot as plt

def create_datas(alpha_min, alpha_max, r, n_elem, delta_x, delta_y, noise):

    theta_min, theta_max = np.radians(alpha_min), np.radians(alpha_max)
    theta = (theta_max - theta_min) * np.random.random(n_elem) + theta_min
    noise_x = np.random.normal(0, noise, n_elem)
    noise_y = np.random.normal(0, noise, n_elem)

    x = r * np.cos(theta) + noise_x + delta_x
    y = r * np.sin(theta) + noise_y + delta_y

    return [x, y, r, delta_x, delta_y]

X, y, r_theo, x_center, y_center = create_datas(alpha_min = 40,
                                                 alpha_max = 150,
                                                 r = 50,
                                                 n_elem = 1000,
                                                 delta_x = 10,
                                                 delta_y = 20,
                                                 noise=3)

mdl = fit_circle.Circle_Regressor(method="leastsq")
X_1 = mdl.fit_transform(X, y)

mdl2 = fit_circle.Circle_Regressor(method="ODR")
X_2 = mdl2.fit_transform(X, y)

mdl3 = fit_circle.Circle_Regressor(method="Alg_approx")
X_3 = mdl3.fit_transform(X, y)

plt.clf()

fig, axes = plt.subplots(3, 3, figsize=(20,20))

axes[0, 0].scatter(X, y, alpha =0.2, s=2)
axes[0, 0].add_patch(plt.Circle((x_center, y_center), r_theo, color='r', ls= "--", lw=3, fill=False, label="Real"))
axes[0, 0].add_patch(plt.Circle(mdl._center, mdl._radius, color='g', ls= "--", lw=2, fill=False, label="Prediction"))
axes[0, 0].scatter(x_center, y_center, color='r', marker="x" )
axes[0, 0].scatter(*mdl._center, color='g', marker="+" )
axes[0, 0].axis('equal')
axes[0, 0].legend()

axes[0, 1].hist(mdl._error, bins=100)
axes[0, 1].set_xlabel("Error", fontsize=10)

axes[0, 2].scatter(X_1[:, 0], X_1[:, 1], alpha =0.2, s=2)
axes[0, 2].set_xlabel("Angle (rad)", fontsize=10)
axes[0, 2].set_ylabel("Radius", fontsize=10)
axes[0, 2].set_ylim(0)

axes[1, 0].scatter(X, y, alpha =0.2, s=2)
axes[1, 0].add_patch(plt.Circle((x_center, y_center), r_theo, color='r', ls= "--", lw=3, fill=False, label="Real"))
axes[1, 0].add_patch(plt.Circle(mdl2._center, mdl2._radius, color='g', ls= "--", lw=2, fill=False, label="Prediction"))
axes[1, 0].scatter(x_center, y_center, color='r', marker="x" )
axes[1, 0].scatter(*mdl2._center, color='g', marker="+" )
axes[1, 0].axis('equal')
axes[1, 0].legend()

axes[1, 1].hist(mdl2._error, bins=100)
axes[1, 1].set_xlabel("Error", fontsize=10)

axes[1, 2].scatter(X_1[:, 0], X_1[:, 1], alpha =0.2, s=2)
axes[1, 2].set_xlabel("Angle (rad)", fontsize=10)
axes[1, 2].set_ylabel("Radius", fontsize=10)
axes[1, 2].set_ylim(0)

axes[2, 0].scatter(X, y, alpha =0.2, s=2)
axes[2, 0].add_patch(plt.Circle((x_center, y_center), r_theo, color='r', ls= "--", lw=3, fill=False, label="Real"))
axes[2, 0].add_patch(plt.Circle(mdl3._center, mdl3._radius, color='g', ls= "--", lw=2, fill=False, label="Prediction"))
axes[2, 0].scatter(x_center, y_center, color='r', marker="x" )
axes[2, 0].scatter(*mdl3._center, color='g', marker="+" )
axes[2, 0].axis('equal')
axes[2, 0].legend()

axes[2, 1].hist(mdl3._error, bins=100)
axes[2, 1].set_xlabel("Error", fontsize=10)

axes[2, 2].scatter(X_1[:, 0], X_1[:, 1], alpha =0.2, s=2)
axes[2, 2].set_xlabel("Angle (rad)", fontsize=10)
axes[2, 2].set_ylabel("Radius", fontsize=10)
axes[2, 2].set_ylim(0)

axes[0, 0].set_title("Scatter Plot of \n noised points", fontsize=20)
axes[0, 1].set_title("Histogram of the error", fontsize=20)
axes[0, 2].set_title("Transformation", fontsize=20)
axes[0, 0].set_ylabel("Method \"leastsq\"", fontsize=20)
axes[1, 0].set_ylabel("Method \"ODR\"", fontsize=20)
axes[2, 0].set_ylabel("Method \"Alg_approx\"", fontsize=20)

plt.savefig("plot_noise.png")
plt.show()