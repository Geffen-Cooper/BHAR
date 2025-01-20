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
        results = [np.array(val) for val in vals]
        results = np.stack(results)
        results_list.append(results)
    
    # stack across seeds
    results_table = np.stack(results_list)
    subject_means = results_table.mean(axis=1)
    seed_std = subject_means.std(axis=0)
    
    return subject_means.mean(axis=0), seed_std


    