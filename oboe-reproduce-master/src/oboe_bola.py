# === updated oboe_bola.py with consistent variable names ===
import numpy as np
import fixed_env as simulator_env
import load_trace
import configmap_bola
import online_changepoint_detection as oncd
from functools import partial
from bola import BOLA

STATE_INFO = 5
HISTORY_LEN = 8

BITRATES_KBPS = [300, 750, 1200, 1850, 2850, 4300]
BUFFER_SCALE = 10.0
KILO = 1000.0
REBUF_WEIGHT = 4.3
SMOOTH_WEIGHT = 1
DEFAULT_LEVEL = 1

OUTPUT_DIR = './test_results'
LOG_PATH = OUTPUT_DIR + '/log_sim_oboebola'

def trim_bandwidth_history(bw_list, max_len):
    if len(bw_list) <= max_len:
        return bw_list, 0
    cutoff = len(bw_list) - max_len
    return bw_list[cutoff:], cutoff

def detect_changepoint(last_change_idx, bw_history, interval=5):
    bw_history, cutoff = trim_bandwidth_history(bw_history, 1000)
    posterior, _ = oncd.online_changepoint_detection(
        np.asarray(bw_history), partial(oncd.constant_hazard, 250), oncd.StudentT(0.1, 0.01, 1, 0))
    interval = min(interval, len(bw_history))
    changes = posterior[interval, interval:-1]
    for i, prob in reversed(list(enumerate(changes))):
        if prob > 0.01 and i + cutoff > last_change_idx and not (i == 0 and last_change_idx > -1):
            return True, i + cutoff
    return False, last_change_idx

def retrieve_gamma(pv_map, mean_bw, std_bw, step=900):
    rounded_mean = int(mean_bw / step) * step
    rounded_std = int(std_bw / step) * step
    search_list = []

    if (rounded_mean, rounded_std) not in pv_map:
        for i in range(2, 1000):
            for delta in [rounded_mean - (i - 1) * step, rounded_mean + (i - 1) * step]:
                for var in range(rounded_std - (i - 1) * step, rounded_std + (i - 1) * step + step, step):
                    if (delta, var) in pv_map:
                        search_list += pv_map[(delta, var)]
            for var in [rounded_std - (i - 1) * step, rounded_std + (i - 1) * step]:
                for delta in range(rounded_mean - (i - 2) * step, rounded_mean + (i - 1) * step, step):
                    if (delta, var) in pv_map:
                        search_list += pv_map[(delta, var)]
            if search_list:
                break
    else:
        search_list = pv_map[(rounded_mean, rounded_std)]

    if not search_list:
        return 1.0, 1.0, 1.0

    return min(search_list), np.mean(search_list), max(search_list)

def main():
    np.random.seed()
    all_times, all_bandwidths, trace_names = load_trace.load_trace('./oboe_traces/')
    net_sim = simulator_env.VideoSimulator(all_times, all_bandwidths)

    log_file = open(LOG_PATH + '_' + trace_names[net_sim.trace_idx], 'w')
    bola_agent = BOLA()

    timestamp = 0
    prev_quality = DEFAULT_LEVEL
    curr_quality = DEFAULT_LEVEL
    throughput_record = []

    video_idx = 0
    last_chd_idx = -1
    gamma_factor = 1.0
    state = np.zeros((STATE_INFO, HISTORY_LEN))

    while True:
        delay, sleep, buf, rebuf, chunk_size, next_chunks, done, remain = \
            net_sim.fetch_chunk(curr_quality)

        timestamp += delay + sleep

        reward = BITRATES_KBPS[curr_quality] / KILO \
               - REBUF_WEIGHT * rebuf \
               - SMOOTH_WEIGHT * abs(BITRATES_KBPS[curr_quality] - BITRATES_KBPS[prev_quality]) / KILO

        prev_quality = curr_quality

        log_file.write(f"{timestamp/KILO:.3f}\t{BITRATES_KBPS[curr_quality]}\t{buf:.2f}\t{rebuf:.2f}\t"
                       f"{chunk_size}\t{delay:.2f}\t{gamma_factor:.2f}\t{reward:.3f}\n")
        log_file.flush()

        state = np.roll(state, -1, axis=1)
        state[0, -1] = BITRATES_KBPS[curr_quality] / max(BITRATES_KBPS)
        state[1, -1] = buf / BUFFER_SCALE
        state[2, -1] = rebuf
        state[3, -1] = chunk_size / delay / KILO
        state[4, -1] = delay / KILO

        throughput_record.append(state[3, -1])
        recent_bw = np.array(throughput_record) * 8000

        chd, chd_idx = detect_changepoint(last_chd_idx, recent_bw)
        if chd:
            last_chd_idx = chd_idx
            recent_segment = recent_bw[chd_idx:]
            mean_bw = np.mean(recent_segment)
            std_bw = np.std(recent_segment)
            _, mean_gamma, _ = retrieve_gamma(configmap_bola.configmap_bola_oboe_900, mean_bw, std_bw)
            gamma_factor = mean_gamma

        curr_quality = bola_agent.run(buf, gamma_factor)

        if done:
            log_file.write('\n')
            log_file.close()

            prev_quality = DEFAULT_LEVEL
            curr_quality = DEFAULT_LEVEL
            state = np.zeros((STATE_INFO, HISTORY_LEN))
            throughput_record = []
            last_chd_idx = -1
            gamma_factor = 1.0

            print("video count", video_idx)
            video_idx += 1

            if video_idx >= len(trace_names):
                break

            log_file = open(LOG_PATH + '_' + trace_names[net_sim.trace_idx], 'w')

if __name__ == '__main__':
    main()
