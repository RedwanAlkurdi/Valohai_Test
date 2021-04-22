import os
import zipfile
import shutil




INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')
TRAIN_IMAGES_DIR = os.path.join(INPUTS_DIR, "training-set-images")
TEST_IMAGES_DIR = os.path.join(INPUTS_DIR, "test-set-images")



# Get the Horse or Human dataset
path_horse_or_human = TRAIN_IMAGES_DIR
# Get the Horse or Human Validation dataset
path_validation_horse_or_human = TEST_IMAGES_DIR




#shutil.rmtree('/tmp')
local_zip = path_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(os.path.join(OUTPUTS_DIR, "training"))
zip_ref.close()

local_zip = path_validation_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(os.path.join(OUTPUTS_DIR, "validation"))
zip_ref.close()


print("Done")

"""
# our example directories and files
train_dir = f'{getcwd()}/tmp/training/train'
validation_dir = f'{getcwd()}/tmp/validation/validation'

train_horses_dir = os.path.join(train_dir, "horses")
train_humans_dir = os.path.join(train_dir, "humans")
validation_horses_dir = os.path.join(validation_dir, "horses")
validation_humans_dir = os.path.join(validation_dir, "humans")

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))
"""
