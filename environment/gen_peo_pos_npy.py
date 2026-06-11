# -*- coding: utf-8 -*-
"""
为 KAIST / NCSU 生成 peo_pos.npy，与 env + subp 观测时序对齐（共 101 帧）。

布局 shape (101, peo_num, 3)，float32：
  · 第 0 帧：reset 后、第一次 env.step 之前 state 中的行人位置，
            即 user_dict 每人首条记录 [t,x,y] 经 core 区域偏移后的 x,y，第三维为 peo_value
            （与 Env.pre_reset 中 _pre_cur_peo_pos_value 一致）
  · 第 1–100 帧：peo_pos_value[0] … peo_pos_value[99]
            （与每次 env.step 开头 self.peo_pos_value[self.step_counter] 一致）

用法（在 code 目录下）:
  python environment/gen_peo_pos_npy.py
  python environment/gen_peo_pos_npy.py KAIST

说明：不 import environment.*.conf，以免拉起 util/torch；下列字段须与对应 conf.py 保持一致。
"""
from __future__ import annotations

import copy
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..'))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import numpy as np

# 与 environment/<DS>/conf.py 中行人相关及坐标字段保持一致（修改 conf 时请同步）
_PE_GEN_CONF = {
    'KAIST': {
        'ref_coordx': 1395,
        'ref_coordy': 997,
        'ref_lon': 127.379589,
        'ref_lat': 36.374767,
        'coordx_per_lon': 89470.20036776649,
        'coordy_per_lat': 111122.19769907878,
        'core_lon_min': 127.355114,
        'core_lat_min': 36.364889,
        'peo_dict_path': 'environment/KAIST/user_dict.npy',
        'peo_pos_value_path': 'environment/KAIST/peo_pos_value.npy',
        'peo_num': 92,
        'peo_value': 1,
        'max_step': 100,
    },
    'NCSU': {
        'ref_coordx': -180,
        'ref_coordy': 935,
        'ref_lon': -78.675481,
        'ref_lat': 35.780790,
        'coordx_per_lon': 90148.59146281201,
        'coordy_per_lat': 111122.19769899677,
        'core_lon_min': -78.687777,
        'core_lat_min': 35.775175,
        'peo_dict_path': 'environment/NCSU/user_dict.npy',
        'peo_pos_value_path': 'environment/NCSU/peo_pos_value.npy',
        'peo_num': 35,
        'peo_value': 1,
        'max_step': 100,
    },
    'purdue197_3': {
        'ref_coordx': 0.0,
        'ref_coordy': 0.0,
        'ref_lon': 0.0,
        'ref_lat': 0.0,
        'coordx_per_lon': 1.0,
        'coordy_per_lat': 1.0,
        'core_lon_min': 0.0,
        'core_lat_min': 0.0,
        'peo_dict_path': 'environment/purdue197_3_48/user_dict.npy',
        'peo_pos_path': 'environment/purdue197_3_48/peo_pos.npy',
        'peo_num': 59,
        'peo_value': 1,
        'max_step': 100,
    },
}


def _lon2coordx(env_conf: dict, lon: float) -> float:
    return env_conf['ref_coordx'] + (lon - env_conf['ref_lon']) * env_conf['coordx_per_lon']


def _lat2coordy(env_conf: dict, lat: float) -> float:
    return env_conf['ref_coordy'] + (lat - env_conf['ref_lat']) * env_conf['coordy_per_lat']


def _resolve_data_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_CODE_ROOT, path.lstrip('./').replace('\\', '/'))


def _load_peo_dict_offset(env_conf: dict) -> dict:
    path = _resolve_data_path(env_conf['peo_dict_path'])
    d = np.load(path, allow_pickle=True)[()]
    d = copy.deepcopy(d)
    ox = _lon2coordx(env_conf, env_conf['core_lon_min'])
    oy = _lat2coordy(env_conf, env_conf['core_lat_min'])
    for peo in d:
        for i in range(len(d[peo])):
            d[peo][i][1] -= ox
            d[peo][i][2] -= oy
    return d


def _pre_cur_peo_pos_value(env_conf: dict, peo_dict: dict) -> np.ndarray:
    P = env_conf['peo_num']
    peo_list = list(peo_dict.keys())[:P]
    out = np.zeros([P, 3], dtype=np.float32)
    for i, peo in enumerate(peo_list):
        out[i, 0] = peo_dict[peo][0][1]
        out[i, 1] = peo_dict[peo][0][2]
        out[i, 2] = env_conf['peo_value']
    return out


def ensure_peo_pos_npy(env_conf: dict) -> str:
    """Generate peo_pos.npy from existing caches only when it is missing."""
    out_path = _resolve_data_path(env_conf['peo_pos_path'])
    if os.path.isfile(out_path):
        return out_path

    peo_dict = _load_peo_dict_offset(env_conf)
    frame0 = _pre_cur_peo_pos_value(env_conf, peo_dict)
    pv_path = _resolve_data_path(env_conf['peo_pos_value_path'])
    peo_pos_value = np.asarray(np.load(pv_path, allow_pickle=True), dtype=np.float32)

    expected_shape = (env_conf['max_step'], env_conf['peo_num'], 3)
    if peo_pos_value.shape != expected_shape:
        raise ValueError(
            f'peo_pos_value.npy expected shape {expected_shape}, '
            f'got {peo_pos_value.shape}')

    peo_pos = np.concatenate([frame0[None, :, :], peo_pos_value], axis=0)
    np.save(out_path, peo_pos.astype(np.float32))
    print(f'[gen_peo_pos_npy] wrote missing {out_path} shape={peo_pos.shape}')
    return out_path


def generate_peo_pos_npy(dataset: str) -> str:
    if dataset not in _PE_GEN_CONF:
        raise KeyError(f'未知数据集 {dataset!r}，可选: {list(_PE_GEN_CONF)}')
    env_conf = _PE_GEN_CONF[dataset]

    peo_dict = _load_peo_dict_offset(env_conf)
    frame0 = _pre_cur_peo_pos_value(env_conf, peo_dict)

    pv_path = env_conf['peo_pos_value_path'].lstrip('./').replace('\\', '/')
    pv_path = os.path.join(_CODE_ROOT, pv_path)
    peo_pos_value = np.asarray(np.load(pv_path, allow_pickle=True), dtype=np.float32)

    T, P, C = peo_pos_value.shape
    if T != env_conf['max_step']:
        raise ValueError(f'{dataset}: peo_pos_value T={T} != max_step={env_conf["max_step"]}')
    if P != env_conf['peo_num']:
        raise ValueError(f'{dataset}: peo_pos_value P={P} != peo_num={env_conf["peo_num"]}')
    if C != 3:
        raise ValueError(f'{dataset}: peo_pos_value last dim {C} != 3')

    peo_pos = np.zeros((101, P, 3), dtype=np.float32)
    peo_pos[0] = frame0
    peo_pos[1:101] = peo_pos_value

    out_dir = os.path.join(_CODE_ROOT, 'environment', dataset)
    out_path = os.path.join(out_dir, 'peo_pos.npy')
    np.save(out_path, peo_pos)
    return out_path


def main() -> None:
    os.chdir(_CODE_ROOT)

    argv = [a for a in sys.argv[1:] if not a.startswith('-')]
    names = argv if argv else ['KAIST', 'NCSU', 'purdue197_3']
    for ds in names:
        path = generate_peo_pos_npy(ds)
        arr = np.load(path)
        print(f'[gen_peo_pos_npy] {ds}: wrote {path} shape={arr.shape} dtype={arr.dtype}')


if __name__ == '__main__':
    main()
