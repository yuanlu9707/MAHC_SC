from .conf import *

class SubLog:
    def __init__(self, ENV_CONF, log_path=None, process_id=0):
        self.ENV_CONF = ENV_CONF
        self.root_path = CONF['root_path']
        self.env_name = self.ENV_CONF['env_name']
        self.log_path = os.path.join(log_path, 'process_' + str(process_id))
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.method_name = CONF['method_name']
        self.episode_info = []

        self.eff_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.f_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.dcr_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.dcr_uav_list = np.zeros([self.ENV_CONF['uav_num'], CONF['train_iter']], dtype=np.float32)
        self.dcr_peo_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.ec_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.ec_uav_list = np.zeros([self.ENV_CONF['uav_num'], CONF['train_iter']], dtype=np.float32)
        self.mec_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.mec_uav_list = np.zeros([self.ENV_CONF['uav_num'], CONF['train_iter']], dtype=np.float32)
        self.cec_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.cec_uav_list = np.zeros([self.ENV_CONF['uav_num'], CONF['train_iter']], dtype=np.float32)
        self.hit_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.hit_uav_list = np.zeros([self.ENV_CONF['uav_num'], CONF['train_iter']], dtype=np.float32)
        self.co_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.co_uav_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.co_peo_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.uav_color = ['C' + str(_) for _ in range(self.ENV_CONF['uav_num'])]
        self.ecr_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        self.charge_list = np.zeros(CONF['train_iter'], dtype=np.float32)
        # 单回合内所有时隙、所有无人机的 reward 之和（与 reward_s 全维求和一致）
        self.episode_total_reward_list = np.zeros(CONF['train_iter'], dtype=np.float32)

    def gen_metrics_result(self, iter_id):
        # 倒数第一行，第一列，状态
        final_st = self.episode_info[-1][0]
        # 环境初始数据量大小
        env_init_poi_value = np.sum(final_st['env_init_poi_value'], axis=-1)
        # 纯人类数据收集量
        pure_peo_dc = final_st['pure_peo_dc']
        # 人类数据收集量
        peo_dc = final_st['cur_peo_dc']
        # 无人机相关状态
        uav_pos_energy_hit_mec_cec_charge = final_st['cur_uav_pos_energy_hit_mec_cec_charge']
        # 无人机数据收集量
        uav_dc = final_st['cur_uav_dc']

        # f 地理公平性
        f = 0.0
        active_poi_id_list = np.nonzero(env_init_poi_value)
        uav_peo_collect_poi_ratio = (peo_dc + np.sum(uav_dc, axis=0))[active_poi_id_list] / env_init_poi_value[
            active_poi_id_list]
        square_of_sum = np.square(np.sum(uav_peo_collect_poi_ratio))
        sum_of_square = np.sum(np.square(uav_peo_collect_poi_ratio))
        if sum_of_square > self.ENV_CONF['min_value']:
            f = square_of_sum / sum_of_square / len(uav_peo_collect_poi_ratio)
        self.f_list[iter_id] = f

        # dcr 人类+无人机 数据收集率
        dcr = (np.sum(peo_dc) + np.sum(uav_dc)) / np.sum(env_init_poi_value)
        self.dcr_list[iter_id] = dcr

        # dcr_uav 无人机数据收集率
        self.dcr_uav_list[:, iter_id] = np.sum(uav_dc, axis=1) / np.sum(env_init_poi_value)

        # dcr_peo 人类数据收集率
        dcr_peo = np.sum(peo_dc) / np.sum(env_init_poi_value)
        self.dcr_peo_list[iter_id] = dcr_peo

        # ec 能耗 移动能耗 + 充电消耗
        ec = np.sum(uav_pos_energy_hit_mec_cec_charge[:, 4]) + np.sum(uav_pos_energy_hit_mec_cec_charge[:, 5])
        self.ec_list[iter_id] = ec

        # ec_uav 无人机能耗集和
        self.ec_uav_list[:, iter_id] = uav_pos_energy_hit_mec_cec_charge[:, 4] + uav_pos_energy_hit_mec_cec_charge[:, 5]

        # mec 总移动能耗
        mec = np.sum(uav_pos_energy_hit_mec_cec_charge[:, 4])
        self.mec_list[iter_id] = mec

        # mec_uav 无人机移动能耗集和
        self.mec_uav_list[:, iter_id] = uav_pos_energy_hit_mec_cec_charge[:, 4]

        # cec 收集数据能耗
        cec = np.sum(uav_pos_energy_hit_mec_cec_charge[:, 5])
        self.cec_list[iter_id] = cec

        # cec_uav 无人机收集数据能耗集和
        self.cec_uav_list[:, iter_id] = uav_pos_energy_hit_mec_cec_charge[:, 5]

        # hit 无人机的碰撞次数
        hit = np.sum(uav_pos_energy_hit_mec_cec_charge[:, 3])
        self.hit_list[iter_id] = hit

        # hit_uav
        self.hit_uav_list[:, iter_id] = uav_pos_energy_hit_mec_cec_charge[:, 3]

        # co_peo 人类利用率
        co_peo = np.sum(peo_dc) / np.sum(pure_peo_dc)
        self.co_peo_list[iter_id] = co_peo

        # co_uav 无人机采集率
        co_uav = (np.sum(uav_dc) - (np.sum(pure_peo_dc) - np.sum(peo_dc))) / (
                np.sum(env_init_poi_value) - np.sum(pure_peo_dc))
        self.co_uav_list[iter_id] = co_uav

        # co 合作因子
        co = (np.sum(uav_dc) + np.sum(peo_dc) - (np.sum(pure_peo_dc) - np.sum(peo_dc))) / np.sum(env_init_poi_value)
        self.co_list[iter_id] = co

        # charge 衡量充电情况
        charge = np.sum(uav_pos_energy_hit_mec_cec_charge[:, 6])
        self.charge_list[iter_id] = charge

        # ecr 用于移动和收集数据能消耗/(总能量+充电量)
        ecr = (np.sum(uav_pos_energy_hit_mec_cec_charge[:, 4]) + np.sum(uav_pos_energy_hit_mec_cec_charge[:, 5])) / (
                self.ENV_CONF['uav_num'] * self.ENV_CONF['uav_init_energy'] + np.sum(
            uav_pos_energy_hit_mec_cec_charge[:, 6]))
        self.ecr_list[iter_id] = ecr
        # eff
        eff = dcr * co * f / (ec / (self.ENV_CONF['uav_num'] * self.ENV_CONF['uav_init_energy'] + np.sum(
            uav_pos_energy_hit_mec_cec_charge[:, 6])))
        self.eff_list[iter_id] = eff

    def record_metrics_result(self):
        self.save_list(self.log_path + '/eff.npy', self.eff_list)
        self.save_list(self.log_path + '/f.npy', self.f_list)
        self.save_list(self.log_path + '/dcr.npy', self.dcr_list)
        self.save_list(self.log_path + '/dcr_peo.npy', self.dcr_peo_list)
        self.save_list(self.log_path + '/ec.npy', self.ec_list)
        self.save_list(self.log_path + '/mec.npy', self.mec_list)
        self.save_list(self.log_path + '/cec.npy', self.cec_list)
        self.save_list(self.log_path + '/hit.npy', self.hit_list)
        self.save_list(self.log_path + '/co.npy', self.co_list)
        self.save_list(self.log_path + '/co_uav.npy', self.co_uav_list)
        self.save_list(self.log_path + '/co_peo.npy', self.co_peo_list)
        self.save_list(self.log_path + '/dcr_uav.npy', self.dcr_uav_list)
        self.save_list(self.log_path + '/ec_uav.npy', self.ec_uav_list)
        self.save_list(self.log_path + '/mec_uav.npy', self.mec_uav_list)
        self.save_list(self.log_path + '/cec_uav.npy', self.cec_uav_list)
        self.save_list(self.log_path + '/hit_uav.npy', self.hit_uav_list)
        self.save_list(self.log_path + '/ecr.npy', self.ecr_list)
        self.save_list(self.log_path + '/charge.npy', self.charge_list)
        self.save_list(self.log_path + '/episode_total_reward.npy', self.episode_total_reward_list)

    def save_list(self, path, result_list):
        np.save(path, result_list)

    def draw_ana(self, iter_id, reward_s):
        r = np.round(np.sum(reward_s[:, 0]), 2)
        dcr = np.round(self.dcr_list[iter_id], 2)
        ec = np.round(self.ec_list[iter_id], 2)
        f = np.round(self.f_list[iter_id], 2)
        eff = np.round(self.eff_list[iter_id], 2)
        charge = np.round(self.charge_list[iter_id], 2)
        title_str = str(iter_id) + ' r=' + str(r) + ' dcr=' + str(
            dcr) + '\n ec=' + str(ec) + ' f=' + str(f) + ' eff=' + str(eff) + ' charge=' + str(charge)

        reward_list = reward_s[:, 0, 0]
        save_path = self.log_path + '/reward_pdf/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig = plt.figure()
        plt.plot(reward_list)
        plt.title('reward ' + title_str)
        Fig.savefig(save_path + 'train_step_' + str(iter_id) + '.png')
        plt.close()

        total_uav_collected_data_ratio_list = np.array(
            [np.sum(info[0]['cur_uav_dc'], axis=1) / np.sum(info[0]['env_init_poi_value']) for info in
             self.episode_info])
        save_path = self.log_path + '/dcr_cdf/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig = plt.figure()
        plt.ylim(ymin=0, ymax=1)
        for uavid in range(self.ENV_CONF['uav_num']):
            plt.plot(total_uav_collected_data_ratio_list[:, uavid], c=self.uav_color[uavid],
                     label='uav_' + str(uavid))
        plt.plot(np.sum(total_uav_collected_data_ratio_list, axis=1), c='black', label='uav_all')
        plt.title('dcr ' + title_str)
        plt.legend()
        Fig.savefig(save_path + 'train_step_' + str(iter_id) + '.png')
        plt.close()
        total_uav_mec_list = np.array([info[0]['cur_uav_pos_energy_hit_mec_cec_charge'][:, 4] for info in
                                       self.episode_info])
        save_path = self.log_path + '/mec_cdf/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig = plt.figure()
        for uavid in range(self.ENV_CONF['uav_num']):
            plt.plot(total_uav_mec_list[:, uavid], c=self.uav_color[uavid],
                     label='uav_' + str(uavid))
        plt.plot(np.mean(total_uav_mec_list, axis=1), c='black', label='uav_all')
        plt.title('mec ' + title_str)
        plt.legend()
        Fig.savefig(save_path + 'train_step_' + str(iter_id) + '.png')
        plt.close()
        total_uav_cec_list = np.array([info[0]['cur_uav_pos_energy_hit_mec_cec_charge'][:, 5] for info in
                                       self.episode_info])
        save_path = self.log_path + '/cec_cdf/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig = plt.figure()
        for uavid in range(self.ENV_CONF['uav_num']):
            plt.plot(total_uav_cec_list[:, uavid], c=self.uav_color[uavid],
                     label='uav_' + str(uavid))
        plt.plot(np.mean(total_uav_cec_list, axis=1), c='black', label='uav_all')
        plt.title('cec ' + title_str)
        plt.legend()
        Fig.savefig(save_path + 'train_step_' + str(iter_id) + '.png')
        plt.close()
        total_uav_e_list = np.array([info[0]['cur_uav_pos_energy_hit_mec_cec_charge'][:, 2] for info in
                                     self.episode_info])
        save_path = self.log_path + '/e_cdf/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig = plt.figure()
        for uavid in range(self.ENV_CONF['uav_num']):
            plt.plot(total_uav_e_list[:, uavid], c=self.uav_color[uavid],
                     label='uav_' + str(uavid))
        plt.plot(np.mean(total_uav_e_list, axis=1), c='black', label='uav_all')
        plt.title('e ' + title_str)
        plt.legend()
        Fig.savefig(save_path + 'train_step_' + str(iter_id) + '.png')
        plt.close()

        total_uav_hit_list = np.array([info[0]['cur_uav_pos_energy_hit_mec_cec_charge'][:, 3] for info in
                                       self.episode_info])
        save_path = self.log_path + '/hit_cdf/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig = plt.figure()
        for uavid in range(self.ENV_CONF['uav_num']):
            plt.plot(total_uav_hit_list[:, uavid], c=self.uav_color[uavid],
                     label='uav_' + str(uavid))
        plt.plot(np.sum(total_uav_hit_list, axis=1), c='black', label='uav_all')
        plt.title('hit ' + title_str)
        plt.legend()
        Fig.savefig(save_path + 'train_step_' + str(iter_id) + '.png')
        plt.close()

    def get_cmap(self, n, name='hsv'):
        return plt.cm.get_cmap(name, n)

    def record_trace_se(self, iter_id, env):
        mpl.style.use('default')
        # Legacy fixed-size canvas:
        # Fig = plt.figure(figsize=(10, 5))
        x_len = float(self.ENV_CONF['field_length'][0])
        y_len = float(self.ENV_CONF['field_length'][1])
        xy_ratio = x_len / max(y_len, 1e-6)
        fig_h = 5.0
        fig_w = max(10.0, fig_h * (2.0 * xy_ratio + 0.8))
        Fig = plt.figure(figsize=(fig_w, fig_h))
        ax1 = Fig.add_subplot(121)
        ax2 = Fig.add_subplot(122)
        cm = plt.cm.get_cmap('RdYlBu_r')
        ax1.set_xlim(xmin=0, xmax=self.ENV_CONF['field_length'][0])
        ax1.set_ylim(ymin=0, ymax=self.ENV_CONF['field_length'][1])
        # ax1.grid(True, linestyle='-.', color='r')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xlim(xmin=0, xmax=self.ENV_CONF['field_length'][0])
        ax2.set_ylim(ymin=0, ymax=self.ENV_CONF['field_length'][1])
        # ax2.grid(True, linestyle='-.', color='r')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')

        # draw peo trace
        peo_x_list = [[] for _ in range(self.ENV_CONF['peo_num'])]
        peo_y_list = [[] for _ in range(self.ENV_CONF['peo_num'])]
        for step_id, info in enumerate(self.episode_info):
            st = info[0]
            for peo_id in range(self.ENV_CONF['peo_num']):
                peo_x_list[peo_id].append(st['cur_peo_pos_value'][peo_id, 0])
                peo_y_list[peo_id].append(st['cur_peo_pos_value'][peo_id, 1])
        for peo_id in range(self.ENV_CONF['peo_num']):
            ax2.scatter(peo_x_list[peo_id], peo_y_list[peo_id], color='black', marker='o', s=0.5)

        # draw uav trace
        u_x_list = [[] for _ in range(self.ENV_CONF['uav_num'])]
        u_y_list = [[] for _ in range(self.ENV_CONF['uav_num'])]
        u_x_list_collect = [[] for _ in range(self.ENV_CONF['uav_num'])]
        u_y_list_collect = [[] for _ in range(self.ENV_CONF['uav_num'])]
        u_x_list_charge = [[] for _ in range(self.ENV_CONF['uav_num'])]
        u_y_list_charge = [[] for _ in range(self.ENV_CONF['uav_num'])]

        for step_id, info in enumerate(self.episode_info):
            st = info[0]
            for uav_id in range(self.ENV_CONF['uav_num']):
                u_x_list[uav_id].append(st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 0])
                u_y_list[uav_id].append(st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 1])
                if st['is_uav_collect'][uav_id]:
                    u_x_list_collect[uav_id].append(st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 0])
                    u_y_list_collect[uav_id].append(st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 1])
                else:
                    u_x_list_charge[uav_id].append(st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 0])
                    u_y_list_charge[uav_id].append(st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 1])

        for uav_id in range(self.ENV_CONF['uav_num']):
            color = self.uav_color[uav_id % len(self.uav_color)]
            ax2.plot(u_x_list[uav_id], u_y_list[uav_id], color=color, linewidth=1.5, alpha=1) #alpha线条透明程度
            ax2.scatter(u_x_list_collect[uav_id], u_y_list_collect[uav_id], color=color, marker='.', s=30, alpha=0.5)
            ax2.scatter(u_x_list_charge[uav_id], u_y_list_charge[uav_id], color=color, marker='+', s=50, )

        # draw blk
        blk_dict = env.blk_dict
        for blk in blk_dict:
            blk_att = blk_dict[blk][0]
            blk_key_list = blk_dict[blk][1]
            if blk_att == 'p':
                pgon1 = plt.Polygon(blk_key_list, color='grey', alpha=0.5)
                ax1.add_patch(pgon1)
                pgon2 = plt.Polygon(blk_key_list, color='grey', alpha=0.5)
                ax2.add_patch(pgon2)
            elif blk_att == 'r':
                circ1 = plt.Circle((blk_key_list[0][0], blk_key_list[0][1]), blk_key_list[1], color='grey',
                                   alpha=0.5)
                ax1.add_patch(circ1)
                circ2 = plt.Circle((blk_key_list[0][0], blk_key_list[0][1]), blk_key_list[1], color='grey',
                                   alpha=0.5)
                ax2.add_patch(circ2)
        # draw charge station
        for xy in env.charge_station_pos:
            circ1 = plt.Circle((xy[0], xy[1]), self.ENV_CONF['charge_sensing_range'], color='blue', fill=False)
            ax1.add_patch(circ1)
            circ2 = plt.Circle((xy[0], xy[1]), self.ENV_CONF['charge_sensing_range'], color='blue', fill=False)
            ax2.add_patch(circ2)

        # draw poi
        p_x = self.episode_info[0][0]['cur_poi_pos_value'][:, 0]
        p_y = self.episode_info[0][0]['cur_poi_pos_value'][:, 1]
        p_v_s = np.sum(self.episode_info[0][0]['cur_poi_pos_value'][:, 2:], axis=-1)
        p_v_e = np.sum(self.episode_info[-1][0]['cur_poi_pos_value'][:, 2:], axis=-1)
        p_v_s_norm = p_v_s / self.ENV_CONF['poi_value_max']
        p_v_e_norm = p_v_e / self.ENV_CONF['poi_value_max']
        ax1.scatter(p_x, p_y, c=p_v_s_norm, vmin=0, vmax=1, cmap=cm, alpha=1, s=10)
        ax2.scatter(p_x, p_y, c=p_v_e_norm, vmin=0, vmax=1, cmap=cm, alpha=1, s=10)
        for x, y in zip(p_x, p_y):
            circ1 = plt.Circle((x, y), self.ENV_CONF['uav_sensing_range'], color='red', alpha=0.01, fill=False)
            ax1.add_patch(circ1)
            circ2 = plt.Circle((x, y), self.ENV_CONF['uav_sensing_range'], color='red', alpha=0.01, fill=False)
            ax2.add_patch(circ2)
            circ1 = plt.Circle((x, y), self.ENV_CONF['peo_sensing_range'], color='grey', alpha=0.01, fill=False)
            ax1.add_patch(circ1)
            circ2 = plt.Circle((x, y), self.ENV_CONF['peo_sensing_range'], color='grey', alpha=0.01, fill=False)
            ax2.add_patch(circ2)

        # eff = np.round(self.eff_list[iter_id], 2)
        # f = np.round(self.f_list[iter_id], 2)
        # dcr = np.round(self.dcr_list[iter_id], 2)
        # dcr_uav = [self.dcr_uav_list[uav_id][iter_id] for uav_id in range(self.ENV_CONF['uav_num'])]
        # dcr_uav = [np.round(np.mean(dcr_uav), 2), np.round(np.var(dcr_uav), 2)]
        # dcr_peo = np.round(self.dcr_peo_list[iter_id], 2)
        # ec = np.round(self.ec_list[iter_id], 2)
        # ec_uav = [self.ec_uav_list[uav_id][iter_id] for uav_id in range(self.ENV_CONF['uav_num'])]
        # ec_uav = [np.round(np.mean(ec_uav), 2), np.round(np.var(ec_uav), 2)]
        # mec = np.round(self.mec_list[iter_id], 2)
        # mec_uav = [self.mec_uav_list[uav_id][iter_id] for uav_id in range(self.ENV_CONF['uav_num'])]
        # mec_uav = [np.round(np.mean(mec_uav), 2), np.round(np.var(mec_uav), 2)]
        # cec = np.round(self.cec_list[iter_id], 2)
        # cec_uav = [self.cec_uav_list[uav_id][iter_id] for uav_id in range(self.ENV_CONF['uav_num'])]
        # cec_uav = [np.round(np.mean(cec_uav), 2), np.round(np.var(cec_uav), 2)]
        # hit = np.round(self.hit_list[iter_id], 2)
        # hit_uav = [self.hit_uav_list[uav_id][iter_id] for uav_id in range(self.ENV_CONF['uav_num'])]
        # hit_uav = [np.round(np.mean(hit_uav), 2), np.round(np.var(hit_uav), 2)]
        # co = np.round(self.co_list[iter_id], 2)
        # co_uav = np.round(self.co_uav_list[iter_id], 2)
        # co_peo = np.round(self.co_peo_list[iter_id], 2)
        #
        # ecr = np.round(self.ecr_list[iter_id], 2)
        # charge = np.round(self.charge_list[iter_id], 2)

        # title_str = 'iter: ' + str(iter_id) \
        #             + ' eff: ' + str(eff) \
        #             + ' f: ' + str(f) \
        #             + ' dcr: ' + str(dcr) \
        #             + ' dcr_uav: ' + str(dcr_uav) \
        #             + ' ec: ' + str(ec) \
        #             + ' mec: ' + str(mec) \
        #             + ' cec: ' + str(cec) \
        #             + ' hit: ' + str(hit) \
        #             + ' co: ' + str(co) \
        #             + '\n' \
        #             + ' co_uav: ' + str(co_uav) \
        #             + ' co_peo: ' + str(co_peo) \
        #             + ' dcr_peo: ' + str(dcr_peo) \
        #             + ' ec_uav: ' + str(ec_uav) \
        #             + '\n' \
        #             + ' mec_uav: ' + str(mec_uav) \
        #             + ' cec_uav: ' + str(cec_uav) \
        #             + ' hit_uav: ' + str(hit_uav) \
        #             + ' ecr: ' + str(ecr) \
        #             + ' charge: ' + str(charge)
        #
        # plt.suptitle(title_str)

        save_path = self.log_path + '/trace_se/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Fig.savefig(save_path + 'iter_id' + str(iter_id) + '.png')
        plt.close()

##################################重画
TRACE_PLOT_BUNDLE_VERSION = 1

def trace_plot_bundle_from_episode_info(episode_info, ENV_CONF):
    """只存 record_trace_se 里两处用到的量，无 obs、无行人轨迹等。

    对应 UAV 段：对每个 step 的 st，需 cur_uav_pos_energy_hit_mec_cec_charge[:,0:2] 与 is_uav_collect
    （原逻辑里按步 append 轨迹线/采集点/充电点）。
    对应 POI 段：首帧 cur_poi_pos_value 的 x,y 与 sum(:,2:)；末帧再 sum(:,2:) 作结束颜色。
    """
    n = len(episode_info)
    uav_num = ENV_CONF['uav_num']
    uav_xy = np.empty((n, uav_num, 2), dtype=np.float32)
    is_uav_collect = np.empty((n, uav_num), dtype=np.bool_)
    for i, info in enumerate(episode_info):
        st = info[0]
        uav_xy[i] = np.asarray(st['cur_uav_pos_energy_hit_mec_cec_charge'][:, :2], dtype=np.float32)
        is_uav_collect[i] = st['is_uav_collect']
    st0 = episode_info[0][0]
    stL = episode_info[-1][0]
    p0 = st0['cur_poi_pos_value']
    poi_xy = np.asarray(p0[:, :2], dtype=np.float32)
    poi_value_sum_start = np.sum(p0[:, 2:], axis=-1).astype(np.float32)
    pL = stL['cur_poi_pos_value']
    poi_value_sum_end = np.sum(pL[:, 2:], axis=-1).astype(np.float32)
    return {
        'version': TRACE_PLOT_BUNDLE_VERSION,
        'uav_xy': uav_xy,  # st['cur_uav_pos_energy_hit_mec_cec_charge'][uav_id, 0:2]
        'is_uav_collect': is_uav_collect,  # st['is_uav_collect']
        'poi_xy': poi_xy,  # [:,0], [:,1]
        'poi_value_sum_start': poi_value_sum_start,  # np.sum(p0[:,2:], -1) 首帧
        'poi_value_sum_end': poi_value_sum_end,  # np.sum(pL[:,2:], -1) 末帧
    }


def plot_trace_bundle_ax2_only(ENV_CONF, env, bundle, save_path):
    """与 record_trace_se 的右子图 ax2 一致：行人轨迹 bundle 中无则跳过；UAV/障碍/充电站/POI(末值色)/感知圈。"""
    if bundle.get('version') != TRACE_PLOT_BUNDLE_VERSION:
        raise ValueError('unsupported trace plot bundle version %r' % (bundle.get('version'),))
    uav_xy = bundle['uav_xy']
    is_col = bundle['is_uav_collect']
    p_x = bundle['poi_xy'][:, 0]
    p_y = bundle['poi_xy'][:, 1]
    p_v_e_norm = bundle['poi_value_sum_end'] / ENV_CONF['poi_value_max']

    mpl.style.use('default')
    # Legacy fixed-size canvas:
    # Fig, ax = plt.subplots(figsize=(5, 5))
    x_len = float(ENV_CONF['field_length'][0])
    y_len = float(ENV_CONF['field_length'][1])
    xy_ratio = x_len / max(y_len, 1e-6)
    fig_h = 5.0
    fig_w = max(5.0, fig_h * xy_ratio)
    Fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cm = plt.cm.get_cmap('RdYlBu_r')
    ax.set_xlim(xmin=0, xmax=ENV_CONF['field_length'][0])
    ax.set_ylim(ymin=0, ymax=ENV_CONF['field_length'][1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')

    n_t, uav_num, _ = uav_xy.shape
    uav_color = ['C' + str(_) for _ in range(uav_num)]
    u_x_list_collect = [[] for _ in range(uav_num)]
    u_y_list_collect = [[] for _ in range(uav_num)]
    u_x_list_charge = [[] for _ in range(uav_num)]
    u_y_list_charge = [[] for _ in range(uav_num)]
    for t in range(n_t):
        for uav_id in range(uav_num):
            x, y = float(uav_xy[t, uav_id, 0]), float(uav_xy[t, uav_id, 1])
            if is_col[t, uav_id]:
                u_x_list_collect[uav_id].append(x)
                u_y_list_collect[uav_id].append(y)
            else:
                u_x_list_charge[uav_id].append(x)
                u_y_list_charge[uav_id].append(y)

    for uav_id in range(uav_num):
        color = uav_color[uav_id % len(uav_color)]
        ax.plot(uav_xy[:, uav_id, 0], uav_xy[:, uav_id, 1], color=color, linewidth=1.5, alpha=1)
        ax.scatter(u_x_list_collect[uav_id], u_y_list_collect[uav_id], color=color, marker='.', s=30, alpha=0.5)
        ax.scatter(u_x_list_charge[uav_id], u_y_list_charge[uav_id], color=color, marker='+', s=50)

    blk_dict = env.blk_dict
    for blk in blk_dict:
        blk_att = blk_dict[blk][0]
        blk_key_list = blk_dict[blk][1]
        if blk_att == 'p':
            ax.add_patch(plt.Polygon(blk_key_list, color='grey', alpha=0.5))
        elif blk_att == 'r':
            ax.add_patch(plt.Circle((blk_key_list[0][0], blk_key_list[0][1]), blk_key_list[1], color='grey',
                                     alpha=0.5))
    for xy in env.charge_station_pos:
        ax.add_patch(plt.Circle((xy[0], xy[1]), ENV_CONF['charge_sensing_range'], color='blue', fill=False))

    ax.scatter(p_x, p_y, c=p_v_e_norm, vmin=0, vmax=1, cmap=cm, alpha=1, s=10)
    for x, y in zip(p_x, p_y):
        ax.add_patch(plt.Circle((x, y), ENV_CONF['uav_sensing_range'], color='red', alpha=0.01, fill=False))
        ax.add_patch(plt.Circle((x, y), ENV_CONF['peo_sensing_range'], color='grey', alpha=0.01, fill=False))

    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    Fig.savefig(save_path, dpi=150)
    plt.close(Fig)
