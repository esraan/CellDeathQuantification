import numpy as np
import pandas as pd
from typing import Union, List, Tuple
import xml.etree.ElementTree as ET


class PhysicellXMLSettingsEditor:
    def __init__(self, xml_full_file_path):
        self.xml_file_path = xml_full_file_path
        self.xml_found = False
        self.settings_xml_tree = None

    def read_xml_file(self):
        """
        reads the xml settings file to an instance attribute
        :return:
        """
        self.settings_xml_tree = ET.parse(self.xml_file_path)
        self.xml_found = True

    def update_single_attribute(self, attribute_key: str, attribute_new_value:Union[str, int, float]):
        """
        inner attribute keys should be concatenated with a '.' between keys, e.g. : <attributeName>.<innerAttributeName>        :param attribute_key:
        :param attribute_key:
        :param attribute_new_value:
        :return:
        """
        assert self.xml_found, 'You must read an xml settings file first'

        root_node = self.settings_xml_tree.getroot()
        tree_path_to_attribute = attribute_key.split('.')
        curr_node = root_node
        for key in tree_path_to_attribute:
            try:
                for child in curr_node:
                    if child.tag.lower() == key.lower():
                        curr_node = child
                        break
            except Exception as e:
                raise ValueError(f'attribute {attribute_key} was not found')
        curr_node.text = str(attribute_new_value)

    def update_multiple_attributes(self, attribute_keys: List[str], attribute_new_values: List[Union[str, int, float]]):
        """
        inner attribute keys should be concatenated with a '.' between keys, e.g. : <attributeName>.<innerAttributeName>        :param attribute_key:
        :param attribute_key:
        :param attribute_new_value:
        :return:
        """

        for single_attribute_key, single_attribute_value in zip(attribute_keys, attribute_new_values):
            self.update_single_attribute(attribute_key=single_attribute_key,
                                         attribute_new_value=single_attribute_value)

    def write_xml_file(self):
        assert self.xml_found, 'You must read an xml settings file first'
        self.settings_xml_tree.write(self.xml_file_path)


class ConfigurationsContainer:
    class ConfigurationObject:
        def __init__(self, tag_name: str,
                     name: str,
                     units: str,
                     id: str, value: str,
                     inner_attributes:List[ConfigurationObject]):
            self.attribute_tag_name = tag_name
            self.attribute_name = name
            self.attribute_units = units
            self.attribute_id = id
            self.attribute_value = value
            self.inner_attributes = inner_attributes

        def __str__(self):
            return "{{0}:}".format(self.attribute_tag_name)

    def __init__(self):
        pass


# TODOs:
# todo: convert the xyt of a single experiment to the appropriate format. including setting the coordinates to a zero centered domain.

# todo: command to copy the 1 custom.cpp 2 custom.h 3 xml settings file 4 main.cpp file
