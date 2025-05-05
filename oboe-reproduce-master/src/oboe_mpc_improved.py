import numpy as np
import fixed_env as simulation_env
import load_trace as trace_loader
import configmap_mpc as config_map
import online_changepoint_detection_improved as changepoint_detector
from functools import partial
import mpc
from online_changepoint_detection_improved import offline_changepoint_detection_pelt

S_INFO = 5
S_LEN = 8

BITRATES = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_SCALE = 10.0
MS_IN_SEC = 1000.0

REBUF_COST = 4.3
SMOOTH_COST = 1
INIT_QUALITY = 1

RESULT_DIR = './test_results'
LOG_FILE = RESULT_DIR + '/log_sim_oboempc_improved'

def trim_bw_history(bw_series, max_len):
    trimmed_series = []
    cutoff_index = 0
    total_len = len(bw_series)
    if total_len <= max_len:
        return bw_series, cutoff_index

    cutoff_index = total_len - max_len
    trimmed_series = bw_series[cutoff_index:]
    return trimmed_series, cutoff_index

def fetch_config_range(perf_vector_map, mean_bw, std_bw, step=900):
    bw_unit = step
    std_unit = step
    rounded_bw = int(float(mean_bw) / bw_unit) * bw_unit
    rounded_std = int(float(std_bw) / std_unit) * std_unit
    candidates = []
    if mean_bw == -1 and std_bw == -1:
        return 0.0, 0.0, 0.0
    if (rounded_bw, rounded_std) not in list(perf_vector_map.keys()):
        for idx in range(2, 1000):
            for bw in [rounded_bw - (idx - 1) * bw_unit, rounded_bw + (idx - 1) * bw_unit]:
                for std in range(rounded_std - (idx - 1) * std_unit, rounded_std + (idx - 1) * std_unit + std_unit, std_unit):
                    if (bw, std) in list(perf_vector_map.keys()):
                        candidates += perf_vector_map[(bw, std)]
            for std in [rounded_std - (idx - 1) * std_unit, rounded_std + (idx - 1) * std_unit]:
                for bw in range(rounded_bw - (idx - 2) * bw_unit, rounded_bw + (idx - 1) * bw_unit, bw_unit):
                    if (bw, std) in list(perf_vector_map.keys()):
                        candidates += perf_vector_map[(bw, std)]
            if candidates:
                break
    else:
        candidates += perf_vector_map[(rounded_bw, rounded_std)]

    if not candidates or max(candidates) == -1.0:
        return 0.0, 0.0, 0.0
    return min(candidates), np.percentile(candidates, 50), max(candidates)

def main():
    np.random.seed()

    cooked_time_list, cooked_bw_list, trace_files = trace_loader.load_trace('./oboe_traces/')
    network_env = simulation_env.Environment(all_cooked_time=cooked_time_list, all_cooked_bw=cooked_bw_list)

    result_path = LOG_FILE + '_' + trace_files[network_env.trace_idx]
    result_file = open(result_path, 'w')

    abr_controller = mpc.mpc()
    current_time = 0

    previous_quality = INIT_QUALITY
    current_quality = INIT_QUALITY

    recent_throughput = []

    videos_done = 0
    last_cp_index = -1
    discount_factor = 0.5
    current_state = np.zeros((S_INFO, S_LEN))

    while True:
        delay, sleep, buffer, rebuffer, chunk_size, next_sizes, finished, chunks_left = network_env.get_video_chunk(current_quality)

        current_time += delay + sleep

        reward = BITRATES[current_quality] / MS_IN_SEC - REBUF_COST * rebuffer - \
                 SMOOTH_COST * np.abs(BITRATES[current_quality] - BITRATES[previous_quality]) / MS_IN_SEC

        previous_quality = current_quality

        result_file.write(f"{current_time / MS_IN_SEC}\t{BITRATES[current_quality]}\t{buffer}\t{rebuffer}\t"
                          f"{chunk_size}\t{delay}\t{discount_factor}\t{reward}\n")
        result_file.flush()

        current_state = np.roll(current_state, -1, axis=1)
        current_state[0, -1] = BITRATES[current_quality] / float(np.max(BITRATES))
        current_state[1, -1] = buffer / BUFFER_SCALE
        current_state[2, -1] = rebuffer
        current_state[3, -1] = float(chunk_size) / float(delay) / MS_IN_SEC
        current_state[4, -1] = float(delay) / MS_IN_SEC

        recent_throughput.append(current_state[3, -1])
        bandwidth_series = np.array(recent_throughput) * 8000.

        cp_found, offset = offline_changepoint_detection_pelt(bandwidth_series[last_cp_index:], penalty=10)
        if cp_found:
            last_cp_index += offset
            tail_bandwidth = bandwidth_series[last_cp_index:]
            avg_bw = np.mean(tail_bandwidth)
            std_bw = np.std(tail_bandwidth)
            _, median_disc, _ = fetch_config_range(config_map.configmap_mpc_oboe_900, avg_bw, std_bw)
            discount_factor = median_disc

        harmonic_bw = np.sum(current_state[3] * current_state[4]) / np.sum(current_state[4])
        future_bw_estimate = harmonic_bw * discount_factor
        current_quality = abr_controller.run(future_bw_estimate * 1000.0, buffer, previous_quality)

        if finished:
            result_file.write('\n')
            result_file.close()

            previous_quality = INIT_QUALITY
            current_quality = INIT_QUALITY
            current_state = np.zeros((S_INFO, S_LEN))
            recent_throughput = []
            last_cp_index = -1
            discount_factor = 0.5

            print("video count", videos_done)
            videos_done += 1

            if videos_done >= len(trace_files):
                break

            result_path = LOG_FILE + '_' + trace_files[network_env.trace_idx]
            result_file = open(result_path, 'w')

if __name__ == '__main__':
    main()
