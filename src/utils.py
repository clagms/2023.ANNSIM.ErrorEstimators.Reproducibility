from src.libraries import *

def local_error(x, ref, shift=0, type=0):
    x = np.array(x)
    ref = np.array(ref)
    tiny = 1e-12
    assert (len(x) == len(ref)), "Dimensions do not match!"
    if type == 0:  # mixed error
        if shift == 0:
            err = np.fabs(x - ref) / (np.fabs(ref) + 1)
        elif shift == 1:
            err = np.fabs(x) / (np.fabs(ref) + 1)
    elif type == 1:  # rms error
        rms = np.zeros(len(x))
        summ = 0
        for i in range(len(x)):
            summ += ref[i] ** 2
            rms[i] = np.sqrt(summ / (i + 1))
        if shift == 0:
            err = np.fabs(x - ref) / (rms + 1)
        elif shift == 1:
            err = np.fabs(x) / (rms + 1)

    return err

# calculate local cumulative error, type -> mathematical norm
def calc_error(x, ref, type=0):
    x = np.array(x)
    ref = np.array(ref)
    if type == 0:
        err = max(np.fabs(x - ref))
    elif type == 1:
        err = max(np.fabs(x - ref) / np.fabs(ref))
    elif type == 2:
        err = np.linalg.norm(x - ref) / np.linalg.norm(ref)
    return err

# calculate relative local error, type is norm again
def calc_local_error(x, ref, type=0):
    x = np.array(x)
    ref = np.array(ref)
    if type == 0:
        err = np.fabs(x - ref)
    elif type == 1:
        err = np.fabs(x - ref) / np.fabs(ref)
    return err

# polynomial extrapolation
def polynomial_extrapolation(y, order=0):
    x = np.zeros(order + 1)
    # assuming constant communication interval size
    for i in range(order + 1):
        x[i] = -1 * i
    y_est = []
    for sys in range(len(y)):
        y_est.append([])
        for var in range(len(y[sys])):
            y_est[sys].append([])
            y_est[sys][var].append(y[sys][var][0])
            y_est[sys][var].append(y[sys][var][0])
            for i in range(2, len(y[sys][var])):
                if i > order:
                    y_window = np.flip(y[sys][var][i - order - 1:i])
                    x_window = x
                else:
                    y_window = np.flip(y[sys][var][0:i])
                    x_window = x[0:i]
                f = interpolate.interp1d(x_window, y_window, kind=order, fill_value="extrapolate")
                y_est[sys][var].append(f([1])[0])
    return y_est