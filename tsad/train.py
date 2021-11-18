import logging
import argparse
import datetime

parser = argparse.ArgumentParser("Train Script")
parser.add_argument("--data", type=str, help="root dir of data")
parser.add_argument("--model", type=str, default="LSTM", help="model type")
parser.add_argument("--verbose", dest="log_level", action="store_const", const=logging.DEBUG,
                    help="debug logging")
parser.set_defaults(log_level=logging.INFO)

args = parser.parse_args()


def get_logger():
    res = logging.getLogger(__name__)
    fp = "{:.0f}.log".format(datetime.datetime.now().timestamp())
    handler = logging.FileHandler(fp, mode='a+', encoding='utf8')
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    res.addHandler(handler)
    res.setLevel(level=args.log_level)
    return res


logger = get_logger()
