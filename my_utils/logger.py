from hpe_library.lib_import import * 

def get_logger(log_dir, num_trial):
    # Generate log file
    log_file = os.path.join(log_dir, 'train_log.txt')
    if os.path.exists(log_file):
        os.remove(log_file)
    train_log = open(os.path.join(log_dir, 'train_log.txt'), 'w') 
    train_log.close()

    # logger
    logger = logging.getLogger(name='MyLog')

    # log format
    formatter_tm = logging.Formatter('[%(asctime)s] %(message)s')

    # clean all handlers
    if logger.hasHandlers(): ## 핸들러 존재 여부
        logger.handlers.clear() ## 핸들러 삭제
    # set log level
    logger.setLevel(logging.INFO)
    # add stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter_tm)
    logger.addHandler(stream_handler)
    # add file handler
    train_log_handler = logging.FileHandler(log_dir + '/train_log.txt'.format(num_trial))
    train_log_handler.setFormatter(formatter_tm)
    logger.addHandler(train_log_handler)

    return logger

def log_configs(logger, args):
    logger.info('# ------------- Configs ------------- #\n')
    logger.info(vars(args))
    for key in vars(args).keys():
        logger.info('{} {}'.format(key, getattr(args, key)))
    logger.info('# --------------------------------- #\n')