import functools
import logging

import psutil

# from memory_profiler import profile as mem_profile

logger = logging.getLogger("dmrghandler")


def print_system_info(message: str = ""):
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage("/")
    logger.info(f"{message}")
    logger.info(f"Memory: {memory_info}")
    logger.info(f"Disk: {disk_info}")


# def mem_tracking(track_mem):
#     def mem_tracking_decorator(func):
#         @functools.wraps(func)
#         def mem_tracking_wrapper(*args, **kwargs):
#             if track_mem:
#                 return mem_profile(func(*args, **kwargs))
#             else:
#                 return func(*args, **kwargs)

#         return mem_tracking_wrapper

#     return mem_tracking_decorator
