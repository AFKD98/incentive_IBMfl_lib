"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""
from enum import Enum


class SerializerTypes(Enum):
    """
    Types of supported Serializers
    """
    PICKLE = 1
    JSON_PICKLE = 2
