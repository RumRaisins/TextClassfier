import logging
from logging import handlers


def create_logger(log_path):
    """
    日志的创建
    :param log_path:
    :return:
    """
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3,
        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    return logger


def calculate_f1(predict_golden):
    tp, fp, fn = 0, 0, 0
    for item in predict_golden:
        predicts = item["predict"]
        labels = item["label"]
        for ele in predicts:
            if ele in labels:
                tp += 1
            else:
                fp += 1
        for ele in labels:
            if ele not in predicts:
                fn += 1
    precision = 1.0 * tp / (tp + fp) if tp + fp else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1