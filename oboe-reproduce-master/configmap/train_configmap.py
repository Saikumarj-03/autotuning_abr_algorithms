import numpy as np
import fixed_env as simulator_env
import load_trace
import mpc

STATE_INFO = 5
HISTORY_LEN = 8
NUM_ACTIONS = 6

BITRATES_KBPS = [300, 750, 1200, 1850, 2850, 4300]
BUFFER_SCALE = 10.0
KILO = 1000.0
REBUF_WEIGHT = 4.3
SMOOTH_WEIGHT = 1
DEFAULT_LEVEL = 1
SEED = 42

def optimize_discount(throughput_mean, throughput_std):
    assert len(BITRATES_KBPS) == NUM_ACTIONS
    trace_times, trace_bw, file_names = load_trace.generate_synthetic_traces(throughput_mean, throughput_std)
    rewards_per_discount = []

    for step in range(10):
        discount = step / 10.0 + 0.1
        env_sim = simulator_env.VideoSimulator(trace_times, trace_bw)
        abr_algo = mpc.mpc()
        time_elapsed = 0

        last_level = DEFAULT_LEVEL
        curr_level = DEFAULT_LEVEL
        episode_rewards = []
        state = np.zeros((STATE_INFO, HISTORY_LEN))
        video_counter = 0

        while True:
            delay, sleep, buffer, rebuf, chunk_size, next_sizes, done, remaining = \
                env_sim.fetch_chunk(curr_level)

            time_elapsed += delay + sleep

            reward = BITRATES_KBPS[curr_level] / KILO \
                   - REBUF_WEIGHT * rebuf \
                   - SMOOTH_WEIGHT * np.abs(BITRATES_KBPS[curr_level] - BITRATES_KBPS[last_level]) / KILO

            episode_rewards.append(reward)
            last_level = curr_level
            state = np.roll(state, -1, axis=1)

            state[0, -1] = BITRATES_KBPS[curr_level] / float(np.max(BITRATES_KBPS))
            state[1, -1] = buffer / BUFFER_SCALE
            state[2, -1] = rebuf
            state[3, -1] = float(chunk_size) / float(delay) / KILO
            state[4, -1] = float(delay) / KILO

            bw_est = np.sum(state[3] * state[4]) / np.sum(state[4]) * 1000.0
            est_future_bw = bw_est * discount
            curr_level = abr_algo.run(est_future_bw, buffer, last_level)

            if done:
                last_level = DEFAULT_LEVEL
                curr_level = DEFAULT_LEVEL
                rewards_per_discount.append(np.sum(episode_rewards))
                episode_rewards.clear()
                state = np.zeros((STATE_INFO, HISTORY_LEN))
                video_counter += 1
                if video_counter >= len(file_names):
                    break

        rewards_per_discount.append(np.array(rewards_per_discount))

    rewards_per_discount = np.array(rewards_per_discount)
    return np.round(np.argmax(rewards_per_discount, axis=0) / 10.0 + 0.1, 1)


if __name__ == '__main__':
    configmap = {}
    for mu_val in range(5, 60):
        for sigma_val in range(1, 40):
            mean_bw = mu_val / 10.0
            std_bw = sigma_val / 10.0
            optimal = optimize_discount(mean_bw, std_bw)
            configmap[(int(mean_bw * 1000), int(std_bw * 1000))] = list(optimal)

    with open('../src/configmap_mpc.py', 'w') as f:
        f.write('configmap_mpc_oboe_900 = ')
        print(configmap, file=f)

    print('done')