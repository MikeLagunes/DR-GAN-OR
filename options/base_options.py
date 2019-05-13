import argparse
import os
import torch
import datetime

now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
checkpoints = './checkpoints'
checkpoints = os.path.join(checkpoints, now)
result = './result'
result = os.path.join(result, now)

datasets = {'hinterstoisser': '/mnt/storage/home/ml15765/scratch/datasets/6D_obj_rec/hinterstoisser',
            'tejani': '/mnt/storage/home/ml15765/scratch/datasets/6D_obj_rec/tejani',
            'tudlight': '/mnt/storage/home/ml15765/scratch/datasets/6D_obj_rec/tudlight',
            'rutgers_apc': '/mnt/storage/home/ml15765/scratch/datasets/6D_obj_rec/rutgers-apc',
            'toyotalight': '/mnt/storage/home/ml15765/scratch/datasets/6D_obj_rec/toyotalight',
            'tless': '/mnt/storage/home/ml15765/scratch/datasets/tless',
            'core50': '/mnt/storage/home/ml15765/scratch/datasets/core50',
            'toybox': '/mnt/storage/home/ml15765/scratch/datasets/toybox'}

class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        self.parser.add_argument('--dataroot', default=datasets['core50'], help='path to images (should have subfolder train and test)')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        self.parser.add_argument('--checkpoints_dir', type=str, default=checkpoints, help='models are saved here')
        self.parser.add_argument('--pretrained_D', type=str, default=None, help='the name of the pretrained discrimiator model to be loaded.')
        self.parser.add_argument('--pretrained_G', type=str, default=None, help='the name of the pretrained generator model to be loaded.')
        self.parser.add_argument('--test_dir', type=str, default=result, help='the dir to save the result')

        self.parser.add_argument('--model', type=str, default='single', help='single/multi')
        self.parser.add_argument('--num_classes', type=int, default=50, help='number of classes on the dataset')
        self.parser.add_argument('--N_z', type=int, default=50, help='the sum of the noise')

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.is_Train = self.is_Train

        # the use of GPU
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if(id >= 0):
                self.opt.gpu_ids.append(id)
        if torch.cuda.is_available() and len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # print the options
        args = vars(self.opt)
        print('-----------------Options----------------')
        for k, v in sorted(args.items()):
            print('{0}: {1}'.format(str(k), str(v)))
        print('-------------------End-------------------')


        result_dir = os.path.join(self.opt.test_dir, self.opt.pretrained_G) if self.opt.pretrained_G else self.opt.test_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # save the options to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.model)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train' if self.opt.is_Train else 'test'))
        with open(file_name, 'wt') as f:
            f.write('-----------------Options-----------------\n')
            for k, v in sorted(args.items()):
                f.write('{0}: {1}\n'.format(str(k), str(v)))
            f.write('-------------------End-------------------\n')

        return self.opt
