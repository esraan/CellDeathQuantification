import os
import unittest
import xml.etree.ElementTree as ET
from PhysicellHTPFramework.SettingsConfigurator import PhysicellXMLSettingsEditor

XML_PATH = '../PhysicellHTPFramework/SAMPLE_FILES_FOR_DEV/PhysiCell_settings.xml'


class PhysicellHTPFrameworkTester(unittest.TestCase):
    def test_read_xml_file(self):
        xml_path = XML_PATH
        PhysicellXMLSettingsEditor(xml_full_file_path=xml_path)
        self.assertEqual(True, True)

    def setUp(self) -> None:
        print('Setting up for tests')
        xml_path = XML_PATH
        self.xml_path = xml_path
        self.settings_editor = PhysicellXMLSettingsEditor(xml_full_file_path=xml_path)
        self.settings_editor.read_xml_file()

    def test_update_single_attribute(self):
        key, new_val = 'domain.x_min', 4
        split_key = key.split('.')
        self.settings_editor.update_single_attribute(attribute_key=key, attribute_new_value=new_val)
        self.assertEqual(self.settings_editor.settings_xml_tree.findall(split_key[0])[0].findall(split_key[1])[0].text, str(new_val))

    def test_update_xml_file(self):
        key, new_val = 'domain.x_min', 8
        split_key = key.split('.')
        self.settings_editor.update_single_attribute(attribute_key=key, attribute_new_value=new_val)
        self.assertEqual(self.settings_editor.settings_xml_tree.findall(split_key[0])[0].findall(split_key[1])[0].text,
                         str(new_val))

        self.settings_editor.write_xml_file()
        xml_tree = ET.parse(self.xml_path)
        self.assertEqual(xml_tree.findall(split_key[0])[0].findall(split_key[1])[0].text, str(new_val))

    def test_multiple_attributes_update(self):
        keys, new_vals = ['domain.x_min', 'domain.x_max', 'domain.y_min', 'domain.y_max'], [200, 200, 200, 200]
        split_keys = [key.split('.') for key in keys]

        self.settings_editor.update_multiple_attributes(attribute_keys=keys, attribute_new_values=new_vals)

        for key_idx, key in enumerate(keys):
            split_key = split_keys[key_idx]
            new_val = new_vals[key_idx]

            self.assertEqual(self.settings_editor.settings_xml_tree.findall(
                split_key[0])[0].findall(split_key[1])[0].text,str(new_val))

        self.settings_editor.write_xml_file()
        xml_tree = ET.parse(self.xml_path)
        for key_idx, key in enumerate(keys):
            split_key = split_keys[key_idx]
            new_val = new_vals[key_idx]
            self.assertEqual(xml_tree.findall(split_key[0])[0].findall(split_key[1])[0].text, str(new_val))

    @classmethod
    def tearDownClass(cls) -> None:
        # print('hey')
        xml_path = XML_PATH
        os.system(f'cp {xml_path[:-4]}_org_copy.xml {xml_path}')

if __name__ == '__main__':
    unittest.main()
