import logging

TIME_FORMAT = "%Y-%m-%d %H:%M:%S,%f"

# We also produce log messages below DEBUG level (machine parsable).
MACHINE_LOG_LEVEL = 5
gama_log = logging.getLogger("gama")
gama_log.setLevel(MACHINE_LOG_LEVEL)
