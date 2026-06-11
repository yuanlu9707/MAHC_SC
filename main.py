from util import *

# 确保工作目录在 code 目录下
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

os.environ['MKL_NUM_THREADS'] = '1'

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('env_name', type=str, help='the name of environment (KAIST or NCSU or purdue)')
    # parser.add_argument('method_name', type=str, help='the name of method (MAPPO_COMM)')
    # parser.add_argument('mode', type=str, help='train or test')
    # args = parser.parse_args()

    args = type('Args', (object,), {
        'env_name': 'KAIST',
        'method_name': 'MAPPO_COMM', #'IPPO', 'MAPPO', 'MAPPO_COMM'
        'mode': 'train' #'train', 'test'
    })()

    ENV_CONF = importlib.import_module('environment.' + args.env_name + '.conf').ENV_CONF
    Env = importlib.import_module('environment.' + args.env_name + '.env').Env

    if args.method_name.startswith('MAPPO'):
        import method.MAPPO.conf as _mappo_conf
        _mappo_conf.CONF['method_name'] = args.method_name
        _mappo_conf.method_name = args.method_name
        importlib.import_module('method.MAPPO.' + args.mode).main(ENV_CONF, Env)
    else:
        import method.IPPO.conf as _ipo_conf
        _ipo_conf.CONF['method_name'] = args.method_name
        _ipo_conf.method_name = args.method_name
        importlib.import_module('method.IPPO.' + args.mode).main(ENV_CONF, Env)


# from util import *

# os.environ['MKL_NUM_THREADS'] = '1'

# if __name__ == '__main__':
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('env_name', type=str, default='NCSU', help='the name of environment (KAIST or NCSU)')
#     # parser.add_argument('method_name', type=str, default='fd_mappo_cubicmap', help='the name of method (fd_mappo_cubicmap)')
#     # parser.add_argument('mode', type=str, default='test', help='train or test')
#     # args = parser.parse_args()

#     # 直接手动指定参数，不再从 sys.argv 读取
#     class Args:
#         env_name = 'KAIST'
#         method_name = 'fd_mappo_cubicmap'
#         mode = 'train'

#     args = Args()

#     ENV_CONF = importlib.import_module('environment.' + args.env_name + '.conf').ENV_CONF
#     Env = importlib.import_module('environment.' + args.env_name + '.env').Env
#     importlib.import_module('method.' + args.method_name + '.' + args.mode).main(ENV_CONF, Env)