# import pandas as pd
# import numpy as np
import argparse
import zipfile
import os
import logging
import zipfile

def unzip_folder(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Folder successfully unzipped!")

parser = argparse.ArgumentParser(description="process LIB result")

# parser.add_argument('--zip_path', type=str, help="path to zip file")

# parser.add_argument('--des_path', type=str, help='path to destination folder')

parser.add_argument('--src', type=str, help='source folder')

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    model_name = args.src
    if os.path.isdir(model_name):
        logging.info(model_name)
        checkpoint_list = os.listdir(os.path.join('.',model_name))
        for i, zip_file_name in enumerate(checkpoint_list):
            des_folder_path = os.path.join('.', model_name, str(i))
            path = os.path.join('.', model_name, zip_file_name)
            # if not os.path.isdir(des_folder_path):
            #     logging.info("Directory not existing. Creating...")
            #     os.mkdir(des_folder_path)
            unzip_folder(path, des_folder_path)

            folder_name = os.listdir(des_folder_path)[0]

            folder_path_list = []
            for path in os.listdir(os.path.join(des_folder_path, folder_name)):
                if os.path.isdir(os.path.join(des_folder_path, folder_name, path)):
                    folder_path_list.append(os.path.join(des_folder_path, folder_name, path))

            folder_path_list.sort(key=lambda x: int(os.path.basename(x)))

            logging.info("folder list length: " + str(len(folder_path_list)))

            with open(os.path.join(des_folder_path, folder_name, 'result.txt'), 'w') as f:
                for folder_path in folder_path_list:
                    file_path = os.path.join(folder_path,'result.txt')
                    with open(file_path, 'r') as pin_f:
                        for line in pin_f:
                            if 'time' in line:
                                continue
                            value = round(float(line.split(':')[1]), 4)
                            f.write(str(value).replace('.',',') + '\t')



