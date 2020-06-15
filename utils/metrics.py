import numpy as np

def PSNR(original, noisy):
    m, n = original.shape

    mse = np.sum(np.square(np.subtract(original, noisy)))
    mse = mse / (m * n)
    return 20 * np.log10(255.0) - 10 * np.log10(mse)