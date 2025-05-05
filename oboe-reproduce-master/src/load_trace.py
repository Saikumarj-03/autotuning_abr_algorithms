# === updated load_trace.py for online stage ===
import os

TRACE_DIRECTORY = './cooked_traces/'

def load_trace(trace_folder=TRACE_DIRECTORY):
    file_names = os.listdir(trace_folder)
    all_time_traces = []
    all_bandwidth_traces = []
    all_trace_names = []

    for filename in file_names:
        file_path = os.path.join(trace_folder, filename)
        time_series = []
        bandwidth_series = []

        with open(file_path, 'rb') as trace_file:
            for line in trace_file:
                tokens = line.split()
                time_series.append(float(tokens[0]))
                bandwidth_series.append(float(tokens[1]))

        all_time_traces.append(time_series)
        all_bandwidth_traces.append(bandwidth_series)
        all_trace_names.append(filename)

    return all_time_traces, all_bandwidth_traces, all_trace_names
