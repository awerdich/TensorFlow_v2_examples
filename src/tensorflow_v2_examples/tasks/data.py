# imports
import os
from luigi import Task

#%% Luigi tasks

class NotMNISTDownload(Task):

    URL = 'https://commondatastorage.googleapis.com/books1000/'
    DATA_ROOT = os.path.koin
