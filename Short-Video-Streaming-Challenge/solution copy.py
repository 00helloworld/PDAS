# # PATH
# # if you want to call your model ,the path setting is

# # NN_MODEL = "/home/team/"$YOUR TEAM NAME"/submit/results/nn_model_ep_18200.ckpt" # model path settings
# import random
# from simulator.video_player import Player
# class Algorithm:
#     def __init__(self):
#         # fill your self params
#         self.buffer_size = 0

#     # Intial
#     def Initialize(self):
#         # Initialize your session or something
#         self.buffer_size = 0

#     # Define your algorithm
#     # The args you can get are as follows:
#     # 1. delay: the time cost of your last operation
#     # 2. rebuf: the length of rebufferment
#     # 3. video_size: the size of the last downloaded chunk
#     # 4. end_of_video: if the last video was ended
#     # 5. play_video_id: the id of the current video
#     # 6. Players: the video data of a RECOMMEND QUEUE of 5 (see specific definitions in readme)
#     # 7. first_step: is this your first step?
#     def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
#         download_video_id = 0
#         bit_rate = 0
#         sleep_time = 500.0
#         download_video_id = random.randint(0,4)
#         bit_rate = random.uniform(0, 2)
#         return download_video_id, bit_rate, sleep_time


import numpy as np
import sys
sys.path.append("..")
from simulator.video_player import BITRATE_LEVELS
from simulator import mpc_module

# 参数设置
MPC_FUTURE_CHUNK_COUNT = 5  # MPC（模型预测控制）算法中考虑的未来视频块数
PAST_BW_LEN = 5             # 记录过去带宽的长度
TAU = 500.0                 # 休眠时间（毫秒），当没有合适的视频块可下载时进入休眠
PLAYER_NUM = 5              # 播放器数量，表示推荐队列中的视频数量
PROLOAD_SIZE = 800000.0     # 每个视频预加载的大小（字节），控制预加载的视频数据量
EPSILON = 3.5               # 用于缓冲区阈值计算的参数
LAMBDA_1 = 0.3              # 用于缓冲区阈值计算的参数，影响带宽因子
LAMBDA_2 = 0.15             # 用于缓冲区阈值计算的参数，影响视频顺序因子
REBUFFER_PENALTY = 4.3      # 重新缓冲的惩罚因子，用于计算QoE
QUALITY_CHANGE_PENALTY = 1  # 质量变化的惩罚因子，用于计算QoE

class Algorithm:
    def __init__(self):
        # 初始化参数
        self.past_bandwidth = np.zeros(PAST_BW_LEN)  # 用于存储过去的带宽记录
        self.past_bandwidth_ests = []                # 用于存储过去的带宽估计
        self.past_errors = []                        # 用于存储带宽估计的误差
        self.sleep_time = 0                          # 休眠时间，当没有块可下载时设定的休眠时长
        self.total_rebuffer_time = 0                 # 总的重新缓冲时间
        self.qoe_score = 0                           # 用于存储当前的QoE分数

    def Initialize(self):
        # 初始化带宽记录和QoE参数
        self.past_bandwidth = np.zeros(PAST_BW_LEN)
        self.total_rebuffer_time = 0
        self.qoe_score = 0

    def estimate_bw(self):
        # 估计未来的带宽
        curr_error = 0
        if len(self.past_bandwidth_ests) > 0 and self.past_bandwidth[-1] != 0:
            # 计算当前估计的误差
            curr_error = abs(self.past_bandwidth_ests[-1] - self.past_bandwidth[-1]) / float(self.past_bandwidth[-1])
        self.past_errors.append(curr_error)

        # 计算过去带宽的调和平均值
        past_bandwidth = self.past_bandwidth[self.past_bandwidth != 0]
        if len(past_bandwidth) == 0:
            harmonic_bandwidth = 0
        else:
            harmonic_bandwidth = len(past_bandwidth) / np.sum(1 / past_bandwidth)

        # 使用过去的误差来调整未来的带宽估计
        max_error = max(self.past_errors[-5:]) if len(self.past_errors) >= 5 else max(self.past_errors)
        future_bandwidth = harmonic_bandwidth / (1 + max_error)

        # 存储带宽估计值
        self.past_bandwidth_ests.append(harmonic_bandwidth)

        return future_bandwidth

    def compute_retention_prob(self, H_i, mc, m):
        # 计算用户保留概率，即在当前播放进度之后继续观看的概率
        if m > mc:
            return float(H_i[m]) / float(H_i[mc])
        return 1.0

    def max_buffer_model(self, p_i_m, T_max_i_m, b_th_i):
        # 计算最大缓冲区阈值，考虑用户保留概率、最大块时间和缓冲区阈值
        return max(p_i_m * T_max_i_m, b_th_i)

    def exponential_buffer_threshold(self, epsilon, lambda_1, lambda_2, C, i, i_c):
        # 计算缓冲区阈值，结合带宽和视频顺序的影响
        return epsilon * np.exp(-lambda_1 * C - lambda_2 * (i - i_c))

    def compute_qoe(self, bit_rate, last_quality):
        # 计算QoE分数，考虑重新缓冲时间和比特率波动
        quality_penalty = abs(bit_rate - last_quality) * QUALITY_CHANGE_PENALTY
        rebuffer_penalty = self.total_rebuffer_time * REBUFFER_PENALTY
        self.qoe_score = bit_rate - quality_penalty - rebuffer_penalty
        return self.qoe_score

    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        # 运行算法，返回下一个要下载的视频ID、比特率和休眠时间

        DEFAULT_QUALITY = 0  # 默认的比特率水平
        if first_step:
            # 如果是第一步，则初始化并返回默认值
            self.sleep_time = 0
            return 0, DEFAULT_QUALITY, self.sleep_time

        if self.sleep_time == 0:
            # 如果不在休眠状态，更新带宽记录
            self.past_bandwidth = np.roll(self.past_bandwidth, -1)
            self.past_bandwidth[-1] = (float(video_size) / 1000000.0) / (float(delay) / 1000.0)  # MB/s

        # 记录重新缓冲时间
        self.total_rebuffer_time += rebuf

        P = []  # 用于存储每个视频未来块的数量
        all_future_chunks_size = []  # 用于存储每个视频在所有比特率下的未来块大小
        future_chunks_highest_size = []  # 用于存储每个视频最高比特率的未来块大小

        for i in range(min(len(Players), PLAYER_NUM)):
            # 遍历推荐队列中的视频
            if Players[i].get_remain_video_num() == 0:
                # 如果视频已经下载完毕，则跳过
                P.append(0)
                all_future_chunks_size.append([0])
                future_chunks_highest_size.append([0])
                continue

            # 计算未来要下载的视频块数量和大小
            P.append(min(MPC_FUTURE_CHUNK_COUNT, Players[i].get_remain_video_num()))
            all_future_chunks_size.append(Players[i].get_undownloaded_video_size(P[-1]))
            future_chunks_highest_size.append(all_future_chunks_size[-1][BITRATE_LEVELS - 1])

        download_video_id = -1  # 用于存储下一个要下载的视频ID
        if Players[0].get_remain_video_num() > 0:
            # 如果当前播放的视频还有未下载的块，则优先下载当前播放的视频
            download_video_id = play_video_id
        else:
            # 否则，按照顺序预加载推荐队列中的视频
            need_loop = True
            cnt = 1
            remain_video_sum = 0
            for seq in range(1, min(len(Players), PLAYER_NUM)):
                remain_video_sum += Players[seq].get_remain_video_num()
            if remain_video_sum == 0:
                need_loop = False
            while need_loop:
                remain_video_sum = 0
                if min(len(Players), PLAYER_NUM) == 1:
                    need_loop = False
                else:
                    for seq in range(1, min(len(Players), PLAYER_NUM)):
                        # 判断是否需要继续预加载
                        if Players[seq].get_preload_size() < (PROLOAD_SIZE * cnt) and Players[seq].get_remain_video_num() > 0:
                            download_video_id = play_video_id + seq
                            need_loop = False
                            break
                        remain_video_sum += Players[seq].get_remain_video_num()
                        if seq == min(len(Players), PLAYER_NUM) - 1:
                            if remain_video_sum > 0:
                                cnt += 1
                            else:
                                need_loop = False

        if download_video_id == -1:
            # 如果没有需要下载的视频，进入休眠状态
            self.sleep_time = TAU
            bit_rate = 0
            download_video_id = play_video_id
        else:
            # 否则，计算下一个视频块的最优比特率
            download_video_seq = download_video_id - play_video_id
            future_bandwidth = self.estimate_bw()  # 估计未来的带宽
            buffer_size = Players[download_video_seq].get_buffer_size()  # 获取当前缓冲区大小
            video_chunk_remain = Players[download_video_seq].get_remain_video_num()  # 获取剩余视频块数
            chunk_sum = Players[download_video_seq].get_chunk_sum()  # 获取视频块总数
            download_chunk_bitrate = Players[download_video_seq].get_downloaded_bitrate()  # 获取已经下载的比特率
            last_quality = DEFAULT_QUALITY
            if len(download_chunk_bitrate) > 0:
                last_quality = download_chunk_bitrate[-1]  # 获取最后一个下载块的比特率

            # 计算用户保留概率和缓冲区阈值
            H_i = Players[download_video_seq].user_retent_rate  # 获取用户保留率数据
            mc = Players[download_video_seq].get_chunk_counter()  # 获取当前块计数器
            T_max_i_m = max([Players[download_video_seq].video_size[br][mc] for br in range(BITRATE_LEVELS)])  # 获取最大块大小
            b_th_i = self.exponential_buffer_threshold(EPSILON, LAMBDA_1, LAMBDA_2, future_bandwidth, download_video_seq, play_video_id)  # 计算缓冲区阈值
            p_i_m = self.compute_retention_prob(H_i, mc, mc + 1)  # 计算保留概率
            b_max_i_m = self.max_buffer_model(p_i_m, T_max_i_m, b_th_i)  # 计算最大缓冲区阈值

            # 在缓冲区小于最大缓冲区阈值时才下载
            if buffer_size <= b_max_i_m:
                bit_rate = mpc_module.mpc(self.past_bandwidth, self.past_bandwidth_ests, self.past_errors, all_future_chunks_size[download_video_seq], P[download_video_seq], buffer_size, chunk_sum, video_chunk_remain, last_quality)
                self.compute_qoe(bit_rate, last_quality)  # 计算QoE分数
                self.sleep_time = 0.0
            else:
                # 否则进入休眠状态
                self.sleep_time = TAU
                bit_rate = 0
                download_video_id = play_video_id

        print('---------', download_video_id, bit_rate, self.sleep_time)
        return download_video_id, bit_rate, self.sleep_time
