import logging

# Create and configure the logger
logging.basicConfig(
    filename='logs.log',
    format="🚀 %(asctime)s - %(levelname)s - %(message)s",
    filemode='w',
    level=logging.INFO
)

# Creating an object for the logger
logger = logging.getLogger()

# console handler to display logs on the CLI
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# log messages in CLI
formatter = logging.Formatter("⚡ %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)