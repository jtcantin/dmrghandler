import logging

import psutil

logger = logging.getLogger("dmrghandler")


def print_system_info(message: str = ""):
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage("/")
    logger.info(f"{message}")
    logger.info(f"Memory: {memory_info}")
    logger.info(f"Disk: {disk_info}")
