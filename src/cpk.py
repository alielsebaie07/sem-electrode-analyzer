import numpy as np


def calculate_cpk(porosity_values, target=30, tolerance=5):
    # Spec limits based on target and tolerance
    USL = target + tolerance  # Upper spec limit
    LSL = target - tolerance  # Lower spec limit

    # Mean and standard deviation of your porosity values
    mu = np.mean(porosity_values)
    sigma = np.std(porosity_values)

    # Avoid division by zero if all values are identical
    if sigma == 0:
        return None

    # Cpk formula
    cpu = (USL - mu) / (3 * sigma)
    cpl = (mu - LSL) / (3 * sigma)
    cpk = min(cpu, cpl)

    return {
        'mean': round(float(mu), 2),
        'std': round(float(sigma), 2),
        'USL': USL,
        'LSL': LSL,
        'Cpk': round(float(cpk), 3),
        'verdict': 'PASS' if cpk >= 1.33 else 'FAIL'
    }
