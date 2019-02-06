import os
import re
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import math
import glob
import pickle

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    """Return True if the file is an image.

    >>> is_image_file('front_1.jpg')
    True
    >>> is_image_file('bs')
    False
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    """Return a list that constains all the image paths in the given dir.

    >>> dir = '/home/jaren/data/train'
    >>> len(make_dataset(dir))
    3566
    >>> dir = '/home/jaren/data/test'
    >>> len(make_dataset(dir))
    2027
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' %dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                id = get_id(path)
                pose = get_pose(path)
                images.append({'path': path,
                                'id': id,
                                'pose': pose,
                                'name': fname})

    return images


def make_dataset2(dir):
    """Return a list that constains all the image paths in the given dir.

    >>> dir = '/home/jaren/data/train'
    >>> len(make_dataset(dir))
    3566
    >>> dir = '/home/jaren/data/test'
    >>> len(make_dataset(dir))
    2027
    """

    if "core50" in dir:

        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' %dir

        train_scenes = "scene_01"
        test_scenes = ["scene_03","scene_07", "scene_10" ]
        filenames = []

        folders = glob.glob(dir + "/*")

        for folder in folders:

            #print(folder)

            folder_id = os.path.split(folder)[1][0:6]

            if folder.find(train_scenes) >= 0: # remove for all scenes

                folder_path = folder + "/*"

                filenames_folder = glob.glob(folder_path)
                filenames_folder.sort()
                filenames.extend(filenames_folder)

            for fname in filenames:
                if is_image_file(fname):
                    path = fname
                    id = get_id(path)
                    pose = get_pose(path)
                    images.append({'path': path,
                                    'id': id,
                                    'pose': pose,
                                    'name': fname})

    elif "6D_obj_rec" in dir:

        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' %dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    id = get_id(path)
                    pose = get_pose(path)
                    images.append({'path': path,
                                    'id': id,
                                    'pose': pose,
                                    'name': fname})
    elif "tless" in dir:

        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' %dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    id = get_id(path)
                    pose = get_pose(path)
                    images.append({'path': path,
                                    'id': id,
                                    'pose': pose,
                                    'name': fname})


    elif "toybox" in dir:

        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' %dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    id = get_id(path)
                    pose = get_pose(path)
                    images.append({'path': path,
                                    'id': id,
                                    'pose': pose,
                                    'name': fname})



    #print(images[1])

    return images


def split_with_same_id(samples):
    """
    split the list samples to sublists that with the same id.
    """
    result = []
    if len(samples)==0:
        return result

    result.append([samples[0]])
    for i in range(1, len(samples)):
        if samples[i-1]['id']==samples[i]['id']:
            result[-1].append(samples[i])
        else:
            result.append([samples[i]])

    return result


def default_loader(path):
    return Image.open(path).convert('RGB')

def get_id_6d(path):
    """Return the id of the image.

    >>> path = '/home/jaren/data/train//001/front_1.jpg'
    >>> get_id(path)
    1
    >>> path = '/home/jaren/data/train//034/front_1.jpg'
    >>> get_id(path)
    34
    """
    #print(path)
    lbl =int(path[-17:-15]) - 1
    #p = re.compile(r'\d{3}')
    return lbl

def get_id(path):
    """Return the id of the image.

    >>> path = '/home/jaren/data/train//001/front_1.jpg'
    >>> get_id(path)
    1
    >>> path = '/home/jaren/data/train//034/front_1.jpg'
    >>> get_id(path)
    34
    """
    #print(path)
    lbl =-1

    if "core50" in path:
        lbl = int(path[-10:-8])-1

    elif "6D_obj" in path:
        lbl =int(path[-17:-15]) - 1

    elif "tless" in path:
        lbl = int(path[-11:-9]) - 1

    elif "toybox" in path:

        obj_class = os.path.split(os.path.split(path)[0])[1]
        obj_class_id = obj_class[0:obj_class.find("_pivot")]

        labels = {}

        with open( "data/labels_toybox.pkl", 'rb') as f:
            labels = pickle.load(f)

        lbl = labels[obj_class_id]

    #p = re.compile(r'\d{3}')
    return lbl

def get_pose(path):
    """Return the pose of the image.
    profile->False
    Frontal->True

    >>> path = '/home/jaren/data/train//001/front_1.jpg'
    >>> get_pose(path)
    True
    >>> path = '/home/jaren/data/train//034/profile_1.jpg'
    >>> get_pose(path)
    False
    """
    p = re.compile(r'front')
    result = True if re.search(p, path) else False
    return result

def show_sample(sample):
    """
    Plot the Tensor sample.
    input: The dict sample of the dataset.
    """
    image = []
    pose = []
    identity = []
    name = []
    for i in range(len(sample)):
        image.append(sample[i]['image'])
        pose.append(sample[i]['pose'])
        identity.append(sample[i]['identity'])
        name.append(sample[i]['name'])
    image = [item for sublist in image for item in sublist]
    pose = [item for sublist in pose for item in sublist]
    identity = [item for sublist in identity for item in sublist]
    name = [item for sublist in name for item in sublist]
    for j in range(len(identity)):

        img = 0.5*image[j] + 0.5
        img = transforms.ToPILImage()(img)

        fig = plt.figure(1)
        ax = fig.add_subplot(4, math.ceil(len(pose)/4), j+1)

        ax.set_title('Frontal:{0}\nIdentity:{1}\nName:{2}'.format(pose[j], identity[j], name[j]))
        plt.imshow(img)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

if __name__ == '__main__':
    dir = '/home/jaren/data//train'
    samples = make_dataset(dir)
    ids = split_with_same_id(samples)