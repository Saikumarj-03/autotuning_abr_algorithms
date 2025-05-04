import numpy as np
import fixed_env as env
import load_trace

S_INFO = 5
S_LEN = 8
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # in Kbps
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
SUMMARY_DIR = './test_results_fcc_1'
LOG_FILE = SUMMARY_DIR + '/log_sim_robustmpc'

class RobustMPC:
    def __init__(self):
        self.past_bandwidths = []
        self.discount_factor = 0.8  # more conservative than standard MPC

    def run(self, buffer_size, last_quality):
        if not self.past_bandwidths:
            est_bw = 3000  # Kbps
        else:
            est_bw = min(self.past_bandwidths) * self.discount_factor

        max_reward = float('-inf')
        best_quality = 0
        for i in range(len(VIDEO_BIT_RATE)):
            rebuf_time = max(0.0, VIDEO_BIT_RATE[i] / est_bw - buffer_size)
            reward = VIDEO_BIT_RATE[i] / M_IN_K - REBUF_PENALTY * rebuf_time - \
                     SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[i] - VIDEO_BIT_RATE[last_quality]) / M_IN_K
            if reward > max_reward:
                max_reward = reward
                best_quality = i

        return best_quality

def main():
    np.random.seed(42)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace('./filtered_mahimahi_traces/')
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    abr_algo = RobustMPC()
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    state = np.zeros((S_INFO, S_LEN))
    video_count = 0

    while True:
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

        time_stamp += delay + sleep_time

        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K - REBUF_PENALTY * rebuf - \
                 SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        last_bit_rate = bit_rate

        log_file.write(f"{time_stamp / M_IN_K:.3f}\t{VIDEO_BIT_RATE[bit_rate]}\t{buffer_size:.3f}\t{rebuf:.3f}\t"
                       f"{video_chunk_size}\t{delay:.3f}\t{reward:.6f}\n")
        log_file.flush()

        state = np.roll(state, -1, axis=1)
        throughput = float(video_chunk_size) / delay / M_IN_K
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = throughput
        state[4, -1] = delay / M_IN_K

        abr_algo.past_bandwidths.append(throughput * 1000)  # Kbps
        if len(abr_algo.past_bandwidths) > 5:
            abr_algo.past_bandwidths.pop(0)

        bit_rate = abr_algo.run(buffer_size, last_bit_rate)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            state = np.zeros((S_INFO, S_LEN))
            abr_algo = RobustMPC()
            time_stamp = 0

            video_count += 1
            print("Video count:", video_count)

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

if __name__ == '__main__':
    main()
