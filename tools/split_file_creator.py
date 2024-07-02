from utility import *
import numpy as np
import random

def _groups_helper(image_folder, groups, label_folder=None, prefix=None, suffix='.npy'):
    if label_folder == None:
        without_label = True
    else:
        without_label = False
    
    image_list = subfiles(image_folder, join=False, prefix=prefix, suffix=suffix)
    image_list.sort(key=lambda x:int(re.findall('\d+', x)[-1]))
    number_of_data = len(image_list)
    print(f"number of data: {number_of_data}")

    random.seed(1111)
    random.shuffle(image_list)

    output_dict = {}
    pointer1 = 0
    for key in groups:
        output_dict[key] = []
        pointer2 = pointer1 + groups[key]
        file_list = image_list[pointer1:pointer2]
        file_list.sort(key=lambda x:int(re.findall('\d+', x)[-1]))
        pointer1 = pointer2
        groups[key] = file_list

    for key in output_dict:    
        for file_name in groups[key]:
            image_name = join(image_folder, file_name)
            assert isfile(image_name), 'An image data does not exist. data name: '+image_name
            if not without_label:
                if file_name[-9:-4] == "_0000":
                    file_name = file_name[:-9] + file_name[-4:]
                if file_name[-12:-7] == "_0000":
                    file_name = file_name[:-12] + file_name[-7:]
                label_name = join(label_folder, file_name)
                
                assert isfile(label_name), 'An label data does not exist. data name: '+label_name
                output_dict[key].append({'image': image_name, 'label': label_name})
            else:
                output_dict[key].append({'image': image_name,})
    return output_dict

def main_random_groups():
    # settings
    dataset_path = './data/source_data'
    split_file_path = './config_files/split_test.json'

    image_folder = join(dataset_path, 'image')
    # label_folder = join(dataset_path, 'label')

    groups = {
        'test': 5,
    }
    outdict1 = _groups_helper(image_folder, groups, label_folder=None, suffix='.nii.gz')

    output_dict = {**outdict1,}
    

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)

if __name__ == "__main__":
    main_random_groups()
