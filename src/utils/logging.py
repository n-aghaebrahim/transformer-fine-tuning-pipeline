import logging

def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )

