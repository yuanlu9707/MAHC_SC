from util import *

dataset_name = 'purdue'
ENV_CONF = {
    # field conf (use local metric coords directly)
    'ref_coordx': 0.0,
    'ref_coordy': 0.0,
    'ref_lon': 0.0,
    'ref_lat': 0.0,
    'coordx_per_lon': 1.0,
    'coordy_per_lat': 1.0,
    'core_lon_min': 0.0,
    'core_lon_max': 1671.8995666382975,
    'core_lat_min': 0.0,
    'core_lat_max': 1221.4710883988212,
    'field_length': [1671.8995666382975, 1221.4710883988212],
    'poi_num': 197,

    # charge station conf
    'charge_station_num': 0,
    'charge_sensing_range': 20,
    'charge_station_dict_path': './environment/' + dataset_name + '/charge_station_dict.npy',

    # poi conf
    'poi_dict_path': './environment/' + dataset_name + '/poi_dict.npy',
    'poi_value_max': 3 * 4,
    'poi_value_min': 2 * 4,

    # peo conf
    'peo_dict_path': './environment/' + dataset_name + '/user_dict.npy',
    'peo_pos_value_path': './environment/' + dataset_name + '/peo_pos_value.npy',
    'peo_pos_path': './environment/' + dataset_name + '/peo_pos.npy',
    'peo_num': 59,
    'peo_value': 1,
    'peo_collect_speed_per_poi': 0.25,
    'peo_sensing_range': 50,
    'record_time_interval': 30,
    'epoch_time_range': 30,

    # block conf
    'blk_dict_path': './environment/' + dataset_name + '/block_dict.npy',

    # uav conf
    'uav_init_pos': 'center',
    'uav_num': 6,
    'uav_collect_speed_per_poi': 5,
    'uav_sensing_range': 60,
    'uav_dis_max': 100.0,
    'uav_speed': 20.0,
    'uav_height': 50.0,
    'uav_init_energy': 100.0,
    'uav_move_energy_consume_ratio': 0.01,
    'uav_collect_energy_consume_ratio': 0.1,

    # other conf
    'positive_factor': 9,
    'penalty_factor': 0.1,
    'min_value': 1e-5,
    'max_step': 100,
    'env_name': dataset_name,
    'dataset_name': dataset_name,
    'data_type_num': 1,
    'charge_factor': 0.0,
    'charge_min_factor': 0.3,
}
