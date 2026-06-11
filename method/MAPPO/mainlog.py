from .conf import *
import matplotlib.pyplot as plt
import shutil

class MainLog:
    v_loss_list = []
    a_loss_list = []
    entropy_list = []
    loss_list = []
    envs_info = {}

    def __init__(self, ENV_CONF):
        self.ENV_CONF = ENV_CONF
        self.CONF = CONF
        self.root_path = CONF['root_path']
        self.env_name = self.ENV_CONF['env_name']
        self.method_name = CONF['method_name']
        # self._time = str(time.strftime("%Y-%m-%d/%H-%M-%S", time.localtime()))
        self._time = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        # self._log_path = os.path.join(self.root_path, self.env_name, self.method_name, self._time)
        self.run_name = f"{self._time}"
        self.run_name += f"_noC4_rate105"
        if self.ENV_CONF['uav_num']!=6:
            self.run_name += f"_{self.ENV_CONF['uav_num']}uav"
        if self.CONF['hidden_size']!=256:
            self.run_name += f"_{self.CONF['hidden_size']}hid"
        if self.CONF['hidden_size_ac']!=128:
            self.run_name += f"_{self.CONF['hidden_size_ac']}ahid"
        # if self.method_name != 'MAPPO':
        #     # if self.CONF['M_size']!=[8, 4, 4]:
        #     #     self.run_name += f"_gru64"
        #     # self.run_name += f"_256gru128toZUtoGNN"
        #     self.run_name += f"_256toZUtoGNN"
        #     # self.run_name += f"_128to64toA"
        if self.method_name in ('MAPPO_CNN', 'MAPPO_CNN_GRU', 'MAPPO_COMM'):
            if self.CONF['cat_fe_mlp']!=False:
                self.run_name += f"_mlp256"
            else:
                if self.CONF['cat_fe_cnn256']!=False:
                    self.run_name += f"_cnn256detach" ##########################detach！！！！！！！！！！！！！
            if self.CONF['cat_fe_gru']!=False:
                self.run_name += f"_z128"

        if self.method_name == 'MAPPO_COMM':
            self.run_name += f"_256to128ZUGNN"#############################################记得改哟！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            # self.run_name += f"_alpha_gate_256to128ZUGNN"
            # self.run_name += f"_256to128ZU_2GNNatt"
            # if self.CONF['hgcn_num_layers']!=2:
            #     self.run_name += f"_{self.CONF['hgcn_num_layers']}h"
            # if self.CONF['hgcn_MHA_num_heads']!=2:
            #     self.run_name += f"_{self.CONF['hgcn_MHA_num_heads']}m"
            if self.CONF['hgcn_node2edge_weight']!=False:
                self.run_name += f"_Wn2e"
            if self.CONF['hgcn_out_dim']!=64:
                self.run_name += f"_F{self.CONF['hgcn_out_dim']}"
            if self.CONF['critic_use_comm']!=False:
                self.run_name += f"_CuseH"
            if self.CONF['comm_clustering_interval']!=10:
                self.run_name += f"_{self.CONF['comm_clustering_interval']}int"
            if self.CONF['comm_stability_threshold']!=0.2:
                self.run_name += f"_{self.CONF['comm_stability_threshold']}th"
            if self.CONF['comm_max_group_size']!=4:
                self.run_name += f"_{self.CONF['comm_max_group_size']}nei"


        if self.CONF['actor_use_pred']!=False:
            self.run_name += f"_pred"
            if self.CONF['pred_out_d']!=64:
                self.run_name += f"_F{self.CONF['pred_out_d']}"
            if self.CONF['critic_use_pred'] != False:
                self.run_name += f"_CuseP"
            if self.CONF['region_cell_num']!=6:
                self.run_name += f"_{self.CONF['region_cell_num']}cell"
            self.run_name += f"_pre{self.CONF['pred_pretrain_epochs']}"
            if self.CONF['pred_pretrain_cosine_schedule'] != False:
                if self.CONF['pred_pretrain_epochs'] > 3000:
                    if self.CONF['pred_lr_late'] == 1e-4:
                        self.run_name += f"1e4"
                    elif self.CONF['pred_lr_late'] == 1e-5:
                        self.run_name += f"1e5"
            if self.CONF['pred_update_each_iter'] != False:
                self.run_name += f"_update"
                if self.CONF['pred_rl_cosine_iters'] != 10000:
                    self.run_name += f"{self.CONF['pred_rl_cosine_iters']}"
                if self.CONF['pred_rl_lr_floor'] == 1e-5:
                    f"1e5"
            if self.CONF['i2c_c2i_use_normalize'] != False:
                self.run_name += f"_l2norm"
            if self.CONF['pred_feat_use_cross_attn'] != True:
                self.run_name += f"_featout"

        self._log_path = os.path.join(self.root_path, self.env_name, self.method_name, self.run_name)
        if not os.path.exists(self._log_path):
            os.makedirs(self._log_path)

    def get_start_time(self):
        return self._time

    def maybe_save_traj_best_episode(self, iter_id, reward_row, last_1k_tracker):
        """最后 1000 个 iter 内：若本 iter 的 8 环境总奖励最大值刷新阶段最优，则拷贝最优子进程的 trace_plot_last.pkl（精简轨迹）到 traj/。"""
        if iter_id < self.CONF['train_iter'] - 1000:
            return
        max_r = float(np.max(reward_row))
        best_pid = int(np.argmax(reward_row))
        if max_r <= last_1k_tracker['best']:
            return
        last_1k_tracker['best'] = max_r
        traj_dir = os.path.join(self._log_path, 'traj')
        os.makedirs(traj_dir, exist_ok=True)
        src = os.path.join(self._log_path, 'process_%d' % best_pid, 'trace_plot_last.pkl')
        dst = os.path.join(traj_dir, 'trace_plot_%d_%d.pkl' % (iter_id, best_pid))
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    def record_report(self, report_str):
        self._report_path = self._log_path + '/report.txt'
        f = open(self._report_path, 'a')
        f.writelines(report_str + '\n')
        f.close()

    def record_env_conf(self):
        conf = self.ENV_CONF
        self._env_conf_path = self._log_path + '/env_conf.txt'
        with open(self._env_conf_path, 'w') as f:
            lines = []
            for k in conf:
                lines.append(str(k) + '\t' + str(conf[k]) + '\n')
            f.writelines(lines)

    def record_conf(self):
        conf = CONF
        self._conf_path = self._log_path + '/conf.txt'
        with open(self._conf_path, 'w') as f:
            lines = []
            for k in conf:
                lines.append(str(k) + '\t' + str(conf[k]) + '\n')
            f.writelines(lines)

    def load_envs_info(self):
        self.envs_info['eff'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'eff.npy'), allow_pickle=True) for env_id in
            range(CONF['env_num'])]

        self.envs_info['f'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'f.npy'), allow_pickle=True) for env_id in
            range(CONF['env_num'])]

        self.envs_info['dcr'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'dcr.npy'), allow_pickle=True) for env_id in
            range(CONF['env_num'])]

        self.envs_info['dcr_peo'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'dcr_peo.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['ec'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'ec.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['mec'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'mec.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['cec'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'cec.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['hit'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'hit.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['co'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'co.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['co_uav'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'co_uav.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['co_peo'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'co_peo.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['ecr'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'ecr.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]

        self.envs_info['charge'] = [
            np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'charge.npy'), allow_pickle=True) for env_id
            in range(CONF['env_num'])]
        etr_vecs = []
        for env_id in range(CONF['env_num']):
            _p = os.path.join(self._log_path, 'process_' + str(env_id), 'episode_total_reward.npy')
            if os.path.isfile(_p):
                etr_vecs.append(np.load(_p, allow_pickle=True))
            else:
                etr_vecs.append(np.zeros_like(self.envs_info['eff'][env_id]))
        self.envs_info['episode_total_reward'] = etr_vecs
        # --------------------------------

        for uav_id in range(self.ENV_CONF['uav_num']):
            self.envs_info['dcr_uav' + str(uav_id)] = [
                np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'dcr_uav.npy'), allow_pickle=True)[
                    uav_id] for env_id in range(CONF['env_num'])]
            self.envs_info['ec_uav' + str(uav_id)] = [
                np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'ec_uav.npy'), allow_pickle=True)[uav_id]
                for env_id in range(CONF['env_num'])]
            self.envs_info['mec_uav' + str(uav_id)] = [
                np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'mec_uav.npy'), allow_pickle=True)[
                    uav_id] for env_id in range(CONF['env_num'])]
            self.envs_info['cec_uav' + str(uav_id)] = [
                np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'cec_uav.npy'), allow_pickle=True)[
                    uav_id] for env_id in range(CONF['env_num'])]
            self.envs_info['hit_uav' + str(uav_id)] = [
                np.load(os.path.join(self._log_path, 'process_' + str(env_id), 'hit_uav.npy'), allow_pickle=True)[
                    uav_id] for env_id in range(CONF['env_num'])]

        for key in self.envs_info:
            self.envs_info[key] = np.concatenate([np.expand_dims(arr, axis=1) for arr in self.envs_info[key]], axis=1)

    def record_metrics_result(self, iter_id):
        np.save(self._log_path + '/eff.npy', self.envs_info['eff'][:iter_id + 1])
        np.save(self._log_path + '/f.npy', self.envs_info['f'][:iter_id + 1])
        np.save(self._log_path + '/dcr.npy', self.envs_info['dcr'][:iter_id + 1])
        np.save(self._log_path + '/dcr_peo.npy', self.envs_info['dcr_peo'][:iter_id + 1])
        np.save(self._log_path + '/ec.npy', self.envs_info['ec'][:iter_id + 1])
        np.save(self._log_path + '/ecr.npy', self.envs_info['ecr'][:iter_id + 1])
        np.save(self._log_path + '/charge.npy', self.envs_info['charge'][:iter_id + 1])
        np.save(self._log_path + '/mec.npy', self.envs_info['mec'][:iter_id + 1])
        np.save(self._log_path + '/cec.npy', self.envs_info['cec'][:iter_id + 1])
        np.save(self._log_path + '/hit.npy', self.envs_info['hit'][:iter_id + 1])
        np.save(self._log_path + '/co.npy', self.envs_info['co'][:iter_id + 1])
        np.save(self._log_path + '/co_uav.npy', self.envs_info['co_uav'][:iter_id + 1])
        np.save(self._log_path + '/co_peo.npy', self.envs_info['co_peo'][:iter_id + 1])
        np.save(
            self._log_path + '/episode_total_reward.npy',
            self.envs_info['episode_total_reward'][:iter_id + 1].astype(np.float32),
        )

        for uav_id in range(self.ENV_CONF['uav_num']):
            np.save(self._log_path + '/dcr_uav' + str(uav_id) + '.npy',
                    self.envs_info['dcr_uav' + str(uav_id)][:iter_id + 1])
            np.save(self._log_path + '/ec_uav' + str(uav_id) + '.npy',
                    self.envs_info['ec_uav' + str(uav_id)][:iter_id + 1])
            np.save(self._log_path + '/mec_uav' + str(uav_id) + '.npy',
                    self.envs_info['mec_uav' + str(uav_id)][:iter_id + 1])
            np.save(self._log_path + '/cec_uav' + str(uav_id) + '.npy',
                    self.envs_info['cec_uav' + str(uav_id)][:iter_id + 1])
            np.save(self._log_path + '/hit_uav' + str(uav_id) + '.npy',
                    self.envs_info['hit_uav' + str(uav_id)][:iter_id + 1])

    def record_reward_train_fig(self, iter_id):
        """阴影：各并行环境在每 iter 上总奖励的 mean±std；实线：均值。保存 reward_train/reward_train_{iter}.png"""
        reward_dir = os.path.join(self._log_path, 'reward_train')
        os.makedirs(reward_dir, exist_ok=True)
        etr = self.envs_info['episode_total_reward'][:iter_id + 1]
        x = np.arange(iter_id + 1)
        y_mean = np.mean(etr, axis=1)
        n_env = etr.shape[1]
        ddof = 1 if n_env > 1 else 0
        y_std = np.std(etr, axis=1, ddof=ddof)
        y_low = y_mean - y_std
        y_high = y_mean + y_std
        fig = plt.figure(figsize=(8, 4))
        plt.fill_between(x, y_low, y_high, alpha=0.35, color='C0', label='%d envs mean ± std' % n_env)
        plt.plot(x, y_mean, color='darkblue', linewidth=2.0, label='mean')
        plt.xlabel('iter')
        plt.ylabel('episode total reward')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(reward_dir, 'reward_train_%d.png' % iter_id), dpi=150)
        plt.close(fig)

    def record_loss(self, v_loss, a_loss, entropy, loss):
        self.v_loss_list.append(v_loss)
        self.a_loss_list.append(a_loss)
        self.entropy_list.append(entropy)
        self.loss_list.append(loss)

        np.save(self._log_path + '/v_loss.npy', self.v_loss_list)
        np.save(self._log_path + '/a_loss.npy', self.a_loss_list)
        np.save(self._log_path + '/entropy.npy', self.entropy_list)
        np.save(self._log_path + '/loss.npy', self.loss_list)

    def save_model(self, model, uid):
        self._model_path = self._log_path + '/model_' + str(uid) + '.pth'
        torch.save(model.state_dict(), self._model_path)
        print('model has been saved to ', self._model_path)

    def save_cur_model(self, model, uid):
        self._model_path = self._log_path + '/tmp_model_' + str(uid) + '.pth'
        torch.save(model.state_dict(), self._model_path)
