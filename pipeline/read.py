import os
import glob

TRAIN_IMAGE_DIR_PATH = 'D://PROJECTS//ADE20K//images//training'

train_image_path = [os.path.join(TRAIN_IMAGE_DIR_PATH, x) for x in glob.iglob(TRAIN_IMAGE_DIR_PATH + '//**//*',
                                                                              recursive=True) if x.endswith('.jpg')]
train_mask_path = [os.path.join(TRAIN_IMAGE_DIR_PATH, x) for x in glob.iglob(TRAIN_IMAGE_DIR_PATH + '//**//*',
                                                                             recursive=True) if x.endswith('_seg.png')]
train_parts_path = [os.path.join(TRAIN_IMAGE_DIR_PATH, x) for x in glob.iglob(TRAIN_IMAGE_DIR_PATH + '//**//*',
                                                                              recursive=True) if
                    x.endswith('_parts_1.png')]

print(len(train_image_path))
print(len(train_mask_path))
print(len(train_parts_path))

VALIDATION_IMAGE_DIR_PATH = 'D://PROJECTS//ADE20K//images//validation'

validation_image_path = [os.path.join(VALIDATION_IMAGE_DIR_PATH, x) for x in
                         glob.iglob(VALIDATION_IMAGE_DIR_PATH + '//**//*',
                                    recursive=True) if x.endswith('.jpg')]
validation_mask_path = [os.path.join(VALIDATION_IMAGE_DIR_PATH, x) for x in
                        glob.iglob(VALIDATION_IMAGE_DIR_PATH + '//**//*',
                                   recursive=True) if
                        x.endswith('_seg.png')]
validation_parts_path = [os.path.join(VALIDATION_IMAGE_DIR_PATH, x) for x in
                         glob.iglob(VALIDATION_IMAGE_DIR_PATH + '//**//*',
                                    recursive=True) if
                         x.endswith('_parts_1.png')]

print(len(validation_image_path))
print(len(validation_mask_path))
print(len(validation_parts_path))
