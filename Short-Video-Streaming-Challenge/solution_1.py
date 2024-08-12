# Begin implementation of the solution.py based on the provided details
import numpy as np
import sys
sys.path.append("..")
from simulator.video_player import BITRATE_LEVELS
from simulator import mpc_module

VIDEO_BIT_RATE = [750,1200,1850]
MPC_FUTURE_CHUNK_COUNT = 5
PAST_BW_LEN = 5
TAU = 500.0  # ms
PLAYER_NUM = 5  

class Algorithm:
    def __init__(self):
        self.past_bandwidth = []
        self.past_bandwidth_ests = []
        self.past_errors = []
        self.buffer_size = 0
        self.last_quality = 0
        self.sleep_time = 0

    def Initialize(self):
        # Initialize the past bandwidth record
        self.past_bandwidth = np.zeros(PAST_BW_LEN)
        self.past_bandwidth_ests = []
        self.past_errors = []
        self.buffer_size = 0
        self.last_quality = 0
        self.sleep_time = 0

    def calculate_retention_prob(self, user_retent_rate, current_chunk):
        # Calculate retention probabilities
        retention_probs = []
        current_rate = float(user_retent_rate[current_chunk])
        for future_rate in user_retent_rate[current_chunk:]:
            retention_probs.append(float(future_rate) / current_rate)
        return retention_probs

    def calculate_max_buffer(self, retention_probs, Players, player, C, play_video_id):
        # Calculate the dynamic maximum buffer size
        T_i_m_max = player.get_video_size(2) / VIDEO_BIT_RATE[2]  # Using the highest bitrate for max buffer calculation
        b_i_th = self.calculate_b_th(Players, player, play_video_id, C)  # Calculate lower bound for max buffer size
        max_buffers = [max(prob * T_i_m_max, b_i_th) for prob in retention_probs]
        return max_buffers

    def calculate_b_th(self, Players, player, play_video_id, C):
        # Calculate the lower bound for the max buffer to avoid rebuffering
        epsilon = 3.5
        lambda_1 = 0.3
        lambda_2 = 0.15
        i_c = play_video_id
        i = Players.index(player)
        distance = i - i_c
        b_th = epsilon * np.exp(-lambda_1 * C - lambda_2 * distance)
        return b_th

    def estimate_bandwidth(self, video_size, delay):
        # Record bandwidth estimation
        self.past_bandwidth = np.roll(self.past_bandwidth, -1)
        self.past_bandwidth[-1] = (float(video_size)/1000000.0) / (float(delay) / 1000.0)  # MB / s
        curr_error = 0
        if len(self.past_bandwidth_ests) > 0 and self.past_bandwidth[-1] != 0:
            curr_error = abs(self.past_bandwidth_ests[-1] - self.past_bandwidth[-1]) / float(self.past_bandwidth[-1])
        self.past_errors.append(curr_error)
        past_bandwidths = self.past_bandwidth[-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        harmonic_bandwidth = 1.0 / (sum(1 / float(val) for val in past_bandwidths) / len(past_bandwidths))
        max_error = max(self.past_errors[-5:])
        future_bandwidth = harmonic_bandwidth / (1 + max_error)
        self.past_bandwidth_ests.append(harmonic_bandwidth)
        return future_bandwidth

    def adjust_for_rebuffer(self, bit_rate, rebuf):
        # Adjust bitrate based on rebuffering
        if rebuf > 0:  # If rebuffering occurred
            bit_rate = max(0, bit_rate - 1)  # Lower the bitrate to avoid further rebuffering
        return bit_rate
    
    def calculate_qoe(self, bit_rate, last_bit_rate, rebuf):
        # Weights
        w1 = 1
        w2 = 1
        w3 = 1.85

        # QoE components
        quality = VIDEO_BIT_RATE[bit_rate]
        quality_variation = abs(quality - VIDEO_BIT_RATE[last_bit_rate])
        qoe = (w1 * quality - 
            w2 * quality_variation - 
            w3 * rebuf)
        return qoe

    def calculate_cost(self, future_chunks_size):
        # Calculate the cost as the sum of future chunk sizes for the selected bitrate
        cost = sum(future_chunks_size) / 1000000.0  # Convert to MB
        return cost

    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        if first_step:
            self.sleep_time = 0
            return 0, 2, self.sleep_time

        # Update bandwidth estimation
        future_bandwidth = self.estimate_bandwidth(video_size, delay)

        # Prepare data for MPC
        P = []
        all_future_chunks_size = []
        for i in range(min(len(Players), PLAYER_NUM)):
            if Players[i].get_remain_video_num() == 0:
                P.append(0)
                all_future_chunks_size.append([0])
                continue
            
            P.append(min(MPC_FUTURE_CHUNK_COUNT, Players[i].get_remain_video_num()))
            all_future_chunks_size.append(Players[i].get_undownloaded_video_size(P[-1]))

        # Calculate retention probabilities and max buffer size for each player
        max_buffers = []
        for player in Players:
            retention_probs = self.calculate_retention_prob(player.user_retent_rate, player.get_play_chunk())
            max_buffer = self.calculate_max_buffer(retention_probs, player, future_bandwidth, play_video_id)
            max_buffers.append(max_buffer)

        download_video_id = -1
        best_score = float('-inf')
        best_bit_rate = 0

        if Players[0].get_remain_video_num() > 0:  # Continue downloading the currently playing video
            download_video_id = play_video_id
        else:  # Preload the next videos in the recommendation queue
            for seq in range(1, min(len(Players), PLAYER_NUM)):
                if Players[seq].get_preload_size() < max_buffers[seq][0] and Players[seq].get_remain_video_num() > 0:
                    download_video_id = play_video_id + seq
                    break

        if download_video_id == -1:
            self.sleep_time = TAU
            bit_rate = 0
            download_video_id = play_video_id
        else:
            download_video_seq = download_video_id - play_video_id
            buffer_size = Players[download_video_seq].get_buffer_size()
            video_chunk_remain = Players[download_video_seq].get_remain_video_num()
            chunk_sum = Players[download_video_seq].get_chunk_sum()
            download_chunk_bitrate = Players[download_video_seq].get_downloaded_bitrate()
            last_quality = self.last_quality
            if len(download_chunk_bitrate) > 0:
                last_quality = download_chunk_bitrate[-1]

            # Iterate over possible bitrates to find the one that maximizes QoE - w4 * Cost
            for bit_rate in range(BITRATE_LEVELS):
                qoe = self.calculate_qoe(bit_rate, last_quality, rebuf)
                cost = self.calculate_cost(all_future_chunks_size[download_video_seq][bit_rate])

                # Calculate the objective: QoE - w4 * Cost
                score = qoe - 0.5 * cost
                if score > best_score:
                    best_score = score
                    best_bit_rate = bit_rate

            bit_rate = best_bit_rate
            self.sleep_time = 0.0

        return download_video_id, bit_rate, self.sleep_time

    
