import numpy as np

def generate_synthetic_traces(mean_bw, std_bw, num_repeats=10):
    time_traces, bw_traces, file_ids = [], [], []
    for i in range(num_repeats):
        time_seq = [t / 2.0 for t in range(600)]
        bw_seq = list(np.clip(np.random.normal(mean_bw, std_bw, 600), 0.1, 100.))

        time_traces.append(time_seq)
        bw_traces.append(bw_seq)
        file_ids.append(str(i))

    return time_traces, bw_traces, file_ids
