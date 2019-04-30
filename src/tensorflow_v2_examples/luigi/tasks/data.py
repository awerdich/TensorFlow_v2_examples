# imports
import os
from luigi import Task, ExternalTask, Parameter
from luigi.contrib.s3 import S3Target
from luigi.format import Nop
from luigi.util import requires
from pdb import set_trace

# local imports
from tensorflow_v2_examples.luigi.target import SuffixPreservingLocalTarget


#%% Luigi tasks

class ContentData(ExternalTask):

    file = Parameter()

    # S3 location
    dataset_root = 's3://notmnist/'

    def output(self):
        dataset_path = os.path.join(self.dataset_root, self.file)
        return S3Target(dataset_path, format = Nop)

class CopyS3DataLocally(Task):

    data_root = os.path.join('data', 'notMNIST')
    dataset = Parameter()

    def requires(self):
        return ContentData(dataset = self.dataset)

    def output(self):
        dataset_path = os.path.join(self.data_root, self.dataset)
        return SuffixPreservingLocalTarget(dataset_path, format = Nop)

    def run(self):

        s3_target = self.requires()

        # Load dataset from S3
        with s3_target.output().open('r') as in_file:


