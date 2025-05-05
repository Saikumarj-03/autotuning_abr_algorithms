import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
BYTES_IN_MB = 1000000.0
BITS_PER_BYTE = 8.0
SEED = 42
CHUNK_DURATION = 4000.0  # in ms
NUM_BITRATES = 6
TOTAL_CHUNKS = 48
MAX_BUFFER_MS = 10.0 * MILLISECONDS_IN_SECOND
BUFFER_DRAIN_STEP = 500.0
PAYLOAD_RATIO = 0.95
ROUND_TRIP_TIME = 80
PACKET_BYTES = 1500
VIDEO_FILE_PREFIX = './envivio/video_size_'

class VideoSimulator:
    def __init__(self, trace_times, trace_bandwidths, seed=SEED):
        assert len(trace_times) == len(trace_bandwidths)
        np.random.seed(seed)

        self.all_times = trace_times
        self.all_bandwidths = trace_bandwidths
        self.chunk_index = 0
        self.playback_buffer = 0

        self.trace_idx = 0
        self.time_trace = self.all_times[self.trace_idx]
        self.bw_trace = self.all_bandwidths[self.trace_idx]

        self.trace_ptr_start = 1
        self.trace_ptr = 1
        self.last_trace_time = self.time_trace[self.trace_ptr - 1]

        self.video_chunk_sizes_dict = {}
        for bitrate in range(NUM_BITRATES):
            self.video_chunk_sizes_dict[bitrate] = []
            with open(VIDEO_FILE_PREFIX + str(bitrate)) as f:
                for line in f:
                    self.video_chunk_sizes_dict[bitrate].append(int(line.split()[0]))

    def fetch_chunk(self, bitrate_level):
        assert 0 <= bitrate_level < NUM_BITRATES
        chunk_size = self.video_chunk_sizes_dict[bitrate_level][self.chunk_index]

        delay_ms = 0.0
        bytes_sent = 0

        while True:
            throughput = self.bw_trace[self.trace_ptr] * BYTES_IN_MB / BITS_PER_BYTE
            duration = self.time_trace[self.trace_ptr] - self.last_trace_time
            payload = throughput * duration * PAYLOAD_RATIO

            if bytes_sent + payload > chunk_size:
                remaining = (chunk_size - bytes_sent) / throughput / PAYLOAD_RATIO
                delay_ms += remaining
                self.last_trace_time += remaining
                break

            bytes_sent += payload
            delay_ms += duration
            self.last_trace_time = self.time_trace[self.trace_ptr]
            self.trace_ptr += 1

            if self.trace_ptr >= len(self.bw_trace):
                self.trace_ptr = 1
                self.last_trace_time = 0

        delay_ms *= MILLISECONDS_IN_SECOND
        delay_ms += ROUND_TRIP_TIME

        rebuffer_time = np.maximum(delay_ms - self.playback_buffer, 0.0)
        self.playback_buffer = np.maximum(self.playback_buffer - delay_ms, 0.0)
        self.playback_buffer += CHUNK_DURATION

        sleep_time = 0
        if self.playback_buffer > MAX_BUFFER_MS:
            excess = self.playback_buffer - MAX_BUFFER_MS
            sleep_time = np.ceil(excess / BUFFER_DRAIN_STEP) * BUFFER_DRAIN_STEP
            self.playback_buffer -= sleep_time

            while True:
                duration = self.time_trace[self.trace_ptr] - self.last_trace_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_trace_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_trace_time = self.time_trace[self.trace_ptr]
                self.trace_ptr += 1
                if self.trace_ptr >= len(self.bw_trace):
                    self.trace_ptr = 1
                    self.last_trace_time = 0

        return_buffer = self.playback_buffer

        self.chunk_index += 1
        chunks_left = TOTAL_CHUNKS - self.chunk_index
        end_of_video = False

        if self.chunk_index >= TOTAL_CHUNKS:
            end_of_video = True
            self.playback_buffer = 0
            self.chunk_index = 0
            self.trace_idx = (self.trace_idx + 1) % len(self.all_times)
            self.time_trace = self.all_times[self.trace_idx]
            self.bw_trace = self.all_bandwidths[self.trace_idx]
            self.trace_ptr = self.trace_ptr_start
            self.last_trace_time = self.time_trace[self.trace_ptr - 1]

        next_chunk_sizes = [self.video_chunk_sizes_dict[i][self.chunk_index] for i in range(NUM_BITRATES)]

        return delay_ms, sleep_time, return_buffer / MILLISECONDS_IN_SECOND, \
            rebuffer_time / MILLISECONDS_IN_SECOND, chunk_size, next_chunk_sizes, \
            end_of_video, chunks_left