import numpy as np
import fixed_env as env
import load_trace
from bola import BOLA

# Constants
S_INFO = 5
S_LEN = 8
A_DIM = 6

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0

REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1

def single_step(mu, sigma):
    gamma_search_space = np.round(np.linspace(0.1, 5.0, 10), 2)


    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(mu, sigma)

    reward_all = []

    for gamma in gamma_search_space:
        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(mu, sigma)

        net_env = env.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw)

        bola_agent = BOLA()
        time_stamp = 0
        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        r_batch = []
        reward_arr = []
        video_count = 0
        state = np.zeros((S_INFO, S_LEN))

        while True:
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay + sleep_time

            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            r_batch.append(reward)
            last_bit_rate = bit_rate

            bit_rate = bola_agent.run(buffer_size, gamma)

            if end_of_video:
                reward_arr.append(np.sum(r_batch))
                del r_batch[:]
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                video_count += 1
                if video_count >= len(all_file_names):
                    break

        reward_all.append(np.array(reward_arr))

    reward_all = np.array(reward_all)  # shape = [len(gamma), num_videos]
    best_gamma_idx = np.argmax(reward_all, axis=0)  # best gamma for each trace
    best_gamma_values = gamma_search_space[best_gamma_idx]

    return np.round(best_gamma_values, 2)  # return best gamma values per trace

def main():
    configmap = {}

    for mu in range(5, 50):      # 0.5 to 6.0 Mbps
        for sigma in range(1, 40):  # 0.1 to 4.0 Mbps std
            mu_ = mu / 10.
            sigma_ = sigma / 10.

            print(f"Processing (mu, sigma) = ({mu_}, {sigma_})")
            best_gammas = single_step(mu_, sigma_)
            configmap[(int(mu_ * 1000), int(sigma_ * 1000))] = [float(x) for x in best_gammas]

    with open('configmap_bola.py', 'w') as f:
        f.write('configmap_bola_oboe_900 = ')
        print(configmap, file=f)

    print('Finished writing configmap_bola.py')

if __name__ == '__main__':
    main()
