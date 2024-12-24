import logging

# Create and configure the logger
logging.basicConfig(
    filename='logs.log',
    format="🚀 %(asctime)s - %(levelname)s - %(message)s",
    filemode='w',
    level=logging.DEBUG  # Set the level to DEBUG to capture all log levels
)

# Creating an object for the logger
logger = logging.getLogger()

# Create a console handler to display logs on the CLI
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter to format the log messages
formatter = logging.Formatter("⚡ %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)