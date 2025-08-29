import logging
import os
from datetime import datetime

# Define the logfile name:
LOGFILE = f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}.log"

#Configure the logging settings:
logging.basicConfig(
    filename=LOGFILE,
    level=logging.DEBUG,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S'
)


# Path to the logfile:
logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOGFILE)