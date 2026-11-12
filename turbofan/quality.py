import numpy as np

def check_prediction_range(predictions, low, high):
    low_count = 0
    high_count = 0
    nan_count = np.isnan(predictions).sum()
  
    for pred in predictions:
        if pred < low:
            low_count += 1

        if pred > high:
            high_count += 1
    if low_count > 0 or high_count > 0 or nan_count > 0:
        raise ValueError(f"{low_count} values are below {low} range and {high_count} values are above {high} range. {nan_count} values are Nan.")
    return True