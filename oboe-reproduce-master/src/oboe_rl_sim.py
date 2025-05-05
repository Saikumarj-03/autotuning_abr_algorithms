import numpy as np
import fixed_env as env
import load_trace
from PensieveAgent import PensieveAgent
import os

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0

SUMMARY_DIR = './test_results'
LOG_FILE = SUMMARY_DIR + '/log_sim_rl'

REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1

S_INFO = 6
S_LEN = 8

def main():
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace('./oboe_traces/')
    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)
    pensieve_agent = PensieveAgent('./models')

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    time_stamp = 0
    video_count = 0
    state = np.zeros((S_INFO, S_LEN))
    last_bit_rate = DEFAULT_QUALITY

    while True:
        bit_rate = pensieve_agent.predict(state)

        delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay + sleep_time
        download_time_ms = delay
        download_time_sec = delay / 1000.0
        throughput_bps = (video_chunk_size * 8.0) / download_time_sec

        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        log_file.write(str(time_stamp / M_IN_K) + '\t' +  # time (s)
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +  # bitrate (kbps)
                       str(buffer_size) + '\t' +  # buffer size (s)
                       str(rebuf) + '\t' +  # rebuffer time (s)
                       str(video_chunk_size) + '\t' +  # chunk size (bytes)
                       str(download_time_ms) + '\t' +  # delay (ms)
                       str(1.0) + '\t' +  # discount (use 1.0 or learned later)
                       str(reward) + '\n')
        log_file.flush()

        state = np.roll(state, -1, axis=1)
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = throughput_bps / M_IN_K / M_IN_K  # in Mbps
        state[4, -1] = delay / M_IN_K
        state[5, -1] = float(np.sum(next_video_chunk_sizes)) / M_IN_K

        last_bit_rate = bit_rate

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            video_count += 1
            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

if __name__ == '__main__':
    main()
