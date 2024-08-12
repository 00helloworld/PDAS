
import numpy as np
import sys
sys.path.append("..")
from simulator.video_player import BITRATE_LEVELS

VIDEO_BIT_RATE = [750,1200,1850]
MPC_FUTURE_CHUNK_COUNT = 5
PAST_BW_LEN = 5
TAU = 500.0  # ms
PLAYER_NUM = 5  

from simulator.video_player import Player
import numpy as np
from simulator.mpc_module import mpc  

class Algorithm:
    def __init__(self):
        # 初始化参数
        self.buffer_size = 0
        self.past_bandwidth = []
        self.past_bandwidth_ests = []
        self.past_errors = []
        # 权重定义
        self.w1 = 1.0
        self.w2 = 1.85
        self.w3 = 0.5
        self.w4 = 0.5
    
    def Initialize(self):
        # 重置初始化
        self.buffer_size = 0
        self.past_bandwidth = []
        self.past_bandwidth_ests = []
        self.past_errors = []

    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        # 1. 更新带宽估计
        self.update_bandwidth_estimate(video_size, delay)
        
        # 2. 计算保留概率和Max Buffer阈值
        retention_probs = self.calculate_retention_probabilities(Players)
        max_buffer_thresholds = self.calculate_max_buffer_thresholds(Players, retention_probs)
        
        # 3. 遍历视频，选择最优的比特率和视频块
        best_video_id, best_bitrate, best_sleep_time = None, None, None
        best_Ui = float('-inf')
        
        for i, player in enumerate(Players):
            if player.get_buffer_size() <= max_buffer_thresholds[i]:
                # 调用K-step RobustMPC来计算最优的比特率选择
                bit_rate = mpc(self.past_bandwidth, self.past_bandwidth_ests, self.past_errors, 
                               player.get_future_video_size(5), 5, player.get_buffer_size(), 
                               player.get_chunk_sum(), player.get_remain_video_num(), last_quality=0)
                
                # 计算总的重缓冲时间
                total_rebuffering_time = self.calculate_total_rebuffering(player, play_video_id, Players, bit_rate)
                
                # 计算当前选择的QoE和Cost，并计算总的U_i
                current_qoe = self.calculate_qoe(bit_rate, retention_probs[i], player, total_rebuffering_time, rebuf)
                current_cost = self.calculate_cost(bit_rate, player)
                current_Ui = current_qoe - self.w4 * current_cost
                
                # 如果当前的选择比之前的好，更新选择
                if current_Ui > best_Ui:
                    best_Ui = current_Ui
                    best_video_id = i
                    best_bitrate = bit_rate
                    best_sleep_time = 0
        
        # 4. 决策输出，如果没有合适的块可供下载，则返回睡眠时间
        if best_video_id is not None:
            return best_video_id, best_bitrate, best_sleep_time
        else:
            return None, None, 500.0  # 睡眠时间设为500ms

    def update_bandwidth_estimate(self, video_size, delay):
        # 根据上次的下载数据更新带宽估计
        estimated_bandwidth = video_size / (delay / 1000.0)
        self.past_bandwidth.append(estimated_bandwidth)
        if len(self.past_bandwidth) > PAST_BW_LEN:
            self.past_bandwidth.pop(0)

    def calculate_retention_probabilities(self, Players):
        # 计算每个视频块的保留概率 p_{i,m}(mc)
        retention_probs = []
        for player in Players:
            # 根据当前播放时间 mc 和用户留存率模型 H_{i,m} 计算
            p_i_m_mc = self.calculate_retention_probability(player)
            retention_probs.append(p_i_m_mc)
        return retention_probs
    
    def calculate_max_buffer_thresholds(self, Players, retention_probs):
        # 计算每个视频的最大缓冲区阈值 b_{i,m}^{max}
        max_buffer_thresholds = []
        for i, player in enumerate(Players):
            max_buffer = self.calculate_max_buffer_threshold(player, retention_probs[i])
            max_buffer_thresholds.append(max_buffer)
        return max_buffer_thresholds
    
    def calculate_retention_probability(self, player):
        # 实现保留概率的计算逻辑
        current_play_time = player.get_play_chunk() * 1000  # 转换为ms
        user_time, user_retent_rate = player.get_user_model()
        if current_play_time >= user_time[-1]:
            return 0.0
        retention_prob = np.interp(current_play_time, user_time, user_retent_rate)
        return retention_prob

    def calculate_max_buffer_threshold(self, player, retention_prob):
        # 实现最大缓冲区阈值的计算逻辑
        max_download_time = player.get_video_len()  
        min_buffer_threshold = 1000.0  # 假设值
        max_buffer_threshold = max(min_buffer_threshold, retention_prob * max_download_time)
        return max_buffer_threshold
    
    def calculate_total_rebuffering(self, player, play_video_id, Players, bit_rate):
        total_rebuffering_time = 0.0
        
        for j in range(play_video_id, len(Players)):
            z = player.get_play_chunk() if j == play_video_id else 0
            pj_zk = self.calculate_retention_probability(Players[j])
            download_time = player.get_video_size(bit_rate) / self.past_bandwidth[-1]  # T(r_{i,m})
            buffer_time = Players[j].get_buffer_size()
            
            # 计算每个视频 j 的重缓冲时间
            rebuffering_time_j = pj_zk * max(download_time - buffer_time, 0)
            
            # 计算从当前视频到视频 j 的切换概率 P_ic,j
            if j > play_video_id:
                pic_j = 1.0
                for l in range(play_video_id, j):
                    pic_j *= (1 - self.calculate_retention_probability(Players[l]))
            else:
                pic_j = 1.0
            
            # 计算加权的重缓冲时间
            total_rebuffering_time += pic_j * rebuffering_time_j
        
        return total_rebuffering_time
    
    def calculate_qoe(self, bit_rate, retention_prob, player, total_rebuffering_time, rebuf):
        # 计算QoE, 结合 Phi, Psi, Gamma
        quality = retention_prob * self.calculate_quality(bit_rate, player)
        smoothness = retention_prob * self.calculate_smoothness(bit_rate, player)
        rebuffering = total_rebuffering_time + rebuf  # 累加即时重缓冲时间
        qoe = self.w1 * quality - self.w2 * smoothness - self.w3 * rebuffering
        return qoe

    def calculate_cost(self, bit_rate, player):
        # 计算下载视频块所需的带宽成本
        video_size = player.get_video_size(bit_rate)  
        cost = video_size / player.get_chunk_sum()  
        return cost
    
    def calculate_quality(self, bit_rate, player):
        return bit_rate  
    
    def calculate_smoothness(self, bit_rate, player):
        last_quality = player.get_downloaded_bitrate()[-1] if player.get_downloaded_bitrate() else bit_rate
        return abs(bit_rate - last_quality)
    
    def calculate_rebuffering_time(self, bit_rate, player):
        download_time = player.get_video_size(bit_rate) / self.past_bandwidth[-1]  
        buffer_time = player.get_buffer_size()
        rebuffering_time = max(0, download_time - buffer_time)
        return rebuffering_time
