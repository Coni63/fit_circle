import scipy
from scipy import optimize, odr
import numpy as np


class Circle_Regressor():
    def __init__(self, method="leastsq"):
        self._radius = None
        self._center = None
        self._error = None
        self.method = method

    def fit(self, X, y):
        if self.method == "leastsq":

            def f_2(c, X, Y):
                """ calculate the variance of the squared distance between the data points and the center of the circle at c=(xc, yc) """
                Ri = (X - c[0]) ** 2 + (Y - c[1]) ** 2
                return Ri - Ri.mean()

            x_m = X.mean()
            y_m = y.mean()

            center_estimate = x_m, y_m
            center, ier = optimize.leastsq(f_2, center_estimate, args=(X, y))

            xc, yc = center
            Ri = np.sqrt((X - xc) ** 2 + (y - yc) ** 2)
            R = Ri.mean()
            self._error = Ri - R
            self._radius = R
            self._center = (xc, yc)

        elif self.method == "Alg_approx":
            x_m = X.mean()
            y_m = y.mean()

            u = X - x_m
            v = y - y_m

            Suv = sum(u * v)
            Suu = sum(u ** 2)
            Svv = sum(v ** 2)
            Suuv = sum(u ** 2 * v)
            Suvv = sum(u * v ** 2)
            Suuu = sum(u ** 3)
            Svvv = sum(v ** 3)

            A = np.array([[Suu, Suv], [Suv, Svv]])
            B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
            uc, vc = scipy.linalg.solve(A, B)

            xc = x_m + uc
            yc = y_m + vc

            Ri = np.sqrt((X - xc) ** 2 + (y - yc) ** 2)
            R = Ri.mean()
            self._error = Ri - R
            self._radius = R
            self._center = (xc, yc)

        elif self.method == "ODR":

            def f_3(beta, x):
                """ return the distance point to estimated circle """
                return (x[0] - beta[0]) ** 2 + (x[1] - beta[1]) ** 2 - beta[2] ** 2

            x_m = X.mean()
            y_m = y.mean()

            R_m = np.sqrt((X - x_m) ** 2 + (y - y_m) ** 2).mean()
            beta0 = [x_m, y_m, R_m]

            lsc_data = odr.Data(np.row_stack([X, y]), y=1)
            lsc_model = odr.Model(f_3, implicit=True)
            lsc_odr = odr.ODR(lsc_data, lsc_model, beta0)
            lsc_out = lsc_odr.run()

            xc, yc, R = lsc_out.beta
            Ri = np.sqrt((X - xc) ** 2 + (y - yc) ** 2)
            self._error = Ri - R
            self._radius = R
            self._center = (xc, yc)
        else:
            print("Unknown method. Must be either leastsq, ODR or Alg_approx")

    def transform(self, X, y):
        theta = np.arctan(X/y)
        r = np.sqrt( (X - self._center[0])**2 + (y - self._center[1])**2 )
        return np.vstack([theta, r]).T

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)