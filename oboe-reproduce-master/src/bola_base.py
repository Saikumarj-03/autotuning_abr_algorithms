import numpy as np
import fixed_env as env
import load_trace
from bola import BOLA

S_INFO = 5
S_LEN = 8

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # in Kbps
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0

REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1

SUMMARY_DIR = './test_results'
LOG_FILE = SUMMARY_DIR + '/log_sim_bolabase'

# Fixed gamma
FIXED_GAMMA = 2.0  # You can experiment with 1.0, 2.0, 3.0 etc.

def main():
    np.random.seed()

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace('./oboe_traces/')

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    bola_agent = BOLA()
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    video_count = 0
    state = np.zeros((S_INFO, S_LEN))

    while True:
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay
        time_stamp += sleep_time

        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        last_bit_rate = bit_rate

        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(FIXED_GAMMA) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        state = np.roll(state, -1, axis=1)
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K
        state[4, -1] = float(delay) / M_IN_K

        # ========== BOLA bitrate selection ==========
        bit_rate = bola_agent.run(buffer_size, FIXED_GAMMA)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            state = np.zeros((S_INFO, S_LEN))

            print("video count", video_count)
            video_count += 1

            if video_count >= len(all_cooked_time):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

if __name__ == '__main__':
    main()
