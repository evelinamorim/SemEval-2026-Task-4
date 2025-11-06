import os
import re

def get_instance_id(file_name:str) -> int:
    """

    :param file_name: a filename in the format of dataset
    :return: the id of that instance based on the pattern of file name
    """
    basename = os.path.split(file_name)[-1]
    instance_id_re = re.compile("\d+")
    match = instance_id_re.match(basename)
    return int(match.group())