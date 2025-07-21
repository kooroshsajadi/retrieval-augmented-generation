import logging

def setup_logging(level=logging.DEBUG, 
                 fmt='%(asctime)s - %(levelname)s - %(message)s'):
    logging.basicConfig(level=level, format=fmt)
