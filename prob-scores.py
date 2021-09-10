import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score, # categorical
    mean_absolute_percentage_error as mape,
    fbeta_score
)

# epsilon = np.finfo(np.float64).eps

def _(y_true, y_pred):
    epsilon = np.finfo(np.float64).eps
    bin_rain = (y_true > 0.) & (y_pred > 0.)
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    mape_rain = ape[bin_rain].mean()
    mape_no_rain = ape[~bin_rain].mean()

    return mape_rain, mape_no_rain

def multi_score_rain(y_true, y_pred):
    # bool rain - not rain
    precise_no_rain = (y_true == 0.) & (y_pred == 0.)
    almost_no_rain = (y_true <= 0.1) & (y_pred <= 0.1) &\
                     (y_true > 0) & (y_pred > 0)
    # precision / recall / f1
    exact_rain = accuracy_score((y_true > 0.).astype(int),
                                (y_pred > 0.).astype(int))
    exact_no_rain = accuracy_score((y_true == 0.).astype(int),
                                   (y_pred == 0.).astype(int))
    precision_rain = precision_score((y_true > 0.).astype(int),
                                     (y_pred > 0.).astype(int))
    precision_no_rain = precision_score((y_true == 0.).astype(int),
                                        (y_pred == 0.).astype(int))
    mape_error = mape(y_true, y_pred)
    fbeta1_rain = fbeta_score((y_true > 0.).astype(int),
                              (y_pred > 0.).astype(int),
                              beta=1.)
    fbeta1_no_rain = fbeta_score((y_true > 0.).astype(int),
                                 (y_pred > 0.).astype(int),
                                 beta=1.)
    fbeta2_rain = fbeta_score((y_true > 0.).astype(int),
                              (y_pred > 0.).astype(int),
                              beta=2.)
    fbeta2_no_rain = fbeta_score((y_true == 0.).astype(int),
                                 (y_pred == 0.).astype(int),
                                 beta=2.)


    return (precise_no_rain.mean(),
            almost_no_rain.mean(),
            exact_rain,
            exact_no_rain,
            precision_rain,
            precision_no_rain,
            mape_error,
            fbeta1_rain,
            fbeta1_no_rain,
            fbeta2_rain,
            fbeta2_no_rain)

x = np.random.rand(10)
x[x < 0.2] = 0.
y = np.arange(10) / 10.

res = _(y, x)

print(np.c_[y, x])
print(res)
