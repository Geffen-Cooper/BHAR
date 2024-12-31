import numpy as np
import pickle
import matplotlib.pyplot as plt

def get_results(result_paths):
    
    results_list = []
    # for each seed
    for result_path in result_paths:
        with open(result_path, 'rb') as f:
            results = pickle.load(f)
        
        # get result for each user
        subjects = results.keys()
        vals = results.values()
        f1s = np.array([val[0] for val in vals])
        results_list.append(f1s)
    
    # stack across seeds
    f1_table = np.stack(results_list)
    subject_means = f1_table.mean(axis=1)
    seed_std = subject_means.std()
    
    return subject_means.mean(), seed_std

    