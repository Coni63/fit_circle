# Regression on circular data

This model is done to be able to do the principle of the linear regression but on circular data. It's using 3 differents methods coming from [scipy](http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html) implemented using sklearn naming.

![rendering](https://github.com/Coni63/fit_circle/blob/master/plot.png)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

No prerequisites are needed, you can download the repository or clone it. 

```
git clone https://github.com/Coni63/circle_fit.git
```

### Installing

No pip installation is available, you just have to include the package in you project folder.


## How to use

The model has been build to be similar to sklearn model. As imple example is available below or in test.py files.

```python
import fit_circle

mdl = fit_circle.Circle_Regressor(method="leastsq")
X_1 = mdl3.fit_transform(X, y)

fig, axes = plt.subplots(figsize=(20,20))

axes.scatter(X, y, alpha =0.2, s=2)
axes.add_patch(plt.Circle(mdl._center, mdl._radius, color='g', ls= "--", lw=2, fill=False, label="Prediction"))
axes.scatter(*mdl._center, color='g', marker="+" )
axes.axis('equal')
axes.legend()

plt.show()
```

The current model provides the following methods :

- fit(X, y) : fit the model on the datas provided (must be 2 vectors of same shape)
- transform(X, y) : transform the data to a (angle, radius) matrix of shape (N_elem, 2)
- fit_transform(X, y) : apply fit then transform

and attributes after fit :

- mdl._radius : an integer with the predicted radius
- mdl._center : an tuple with the predicted center
- mdl._error :  a vector of (N_pts,) with all errors

## Results

A more detailed explanation of the model is available in the Notebook Fit_Circle. As it try to minimize the distance point to circle, you may end up with quite different prediction compare to the initial circle as you can see below.

![rendering](https://github.com/Coni63/fit_circle/blob/master/plot_noise.png)

This is related to the arc and noise of the data.


## Contributor

* **Nicolas MINE** - *Initial work* - [Coni63](https://github.com/Coni63)


## Acknowledgments

There is no checks / error handling implemented yet.
You should not provide a vectors with missing values. 