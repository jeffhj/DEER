from time import sleep
from typing import Iterable, List
import numpy as np
from heapq import nlargest, nsmallest
from multiprocessing import Pool
from threading import Thread
from math import ceil
import subprocess
import json
import pickle
import csv
import tqdm
import time
from collections import defaultdict

def calculate_time(func):
     
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def inner1(*args, **kwargs):
 
        # storing time before function execution
        begin = time.time()
         
        data = func(*args, **kwargs)
 
        # storing time after function execution
        end = time.time()
        print("Total time taken in : ", func.__name__, end - begin)
        return data
 
    return inner1

def ugly_normalize(vecs:np.ndarray):
    normalizers = np.sqrt((vecs * vecs).sum(axis=1))
    normalizers[normalizers==0]=1
    return (vecs.T / normalizers).T

def simple_normalize(vec:np.ndarray):
    normalizer = np.sqrt(np.matmul(vec, vec))
    if normalizer == 0:
        normalizer = 1
    return vec / normalizer

def ntopidx(n, score:Iterable):
    s = nlargest(n, zip(np.arange(len(score)), score), key = lambda x: x[1])
    return [item[0] for item in s]

def nsmallidx(n, score:Iterable):
    s = nsmallest(n, zip(np.arange(len(score)), score), key = lambda x: x[1])
    return [item[0] for item in s]

def my_read(file_name:str):
    return open(file_name, 'r').read().strip().split('\n')

def my_read_pickle(file_name:str):
    with open(file_name, 'rb') as f_in:
        return pickle.load(f_in)

def my_write(file_name:str, content:List[str]):
    with open(file_name, 'w') as f_out:
        f_out.write('\n'.join(content))
        
def my_write_pickle(file_name:str, content):
    with open(file_name, 'wb') as f_out:
        pickle.dump(content, f_out)

def my_json_read(file_name:str):
    return json.load(open(file_name, 'r'))

def my_csv_read(file_name:str, delimiter:str=','):
    return csv.reader(open(file_name, 'r'), delimiter=delimiter)
    
class MyMultiProcessing:
    def __init__(self, thread_num:int=1):
        self.thread_num = thread_num

    @calculate_time
    def run(self, func, input_list:list):
        print('Number of lines is %d' % len(input_list))
        with Pool(self.thread_num) as pool:
            print('Start assigning jobs')
            jobs = [pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func, args=(argument,)) for argument in input_list]
            pool.close()
            print('Jobs are assigned, start getting results')
            return [job.get() for job in tqdm.tqdm(jobs)]

class MultiThreading:
    def __line_process_wrapper(self, line_operation, input_list:list, output_list:list):
        for line in input_list:
            result = line_operation(line)
            if result is not None:
                output_list.append(result)
            sleep(0.1)

    def run(self, line_operation, input_list:list, thread_num:int=1):
        if thread_num <= 0:
            return
        line_count = len(input_list)
        print('Number of lines is %d' % (line_count))
        unit_lines = ceil(line_count / thread_num)

        result = [[] for i in range(thread_num)]
        threads = [Thread(target=self.__line_process_wrapper, args=(line_operation, input_list[unit_lines*i : unit_lines*(i+1)], result[i])) for i in range(thread_num)]

        for i in range(thread_num):
            threads[i].setDaemon(True)
            threads[i].start()

        for i in range(thread_num):
            threads[i].join()
        ret = []
        for sub in result:
            ret += sub
        return ret

def my_email(title:str, message:str, email:str):
    cmd = 'echo "%s" | mail -s "%s" %s' % (message, title, email)
    subprocess.Popen(cmd, shell=True)

# Function for batch generation
def batch(sents:Iterable, n:int):
    l = len(sents)
    for ndx in range(0, l, n):
        yield sents[ndx:min(ndx + n, l)]

class SparseRetrieveSentForPairCoOccur:
    def __init__(self, sent_file:str, occur_file:str):
        self._sents = my_read(sent_file)
        self._occur_dict = defaultdict(set)
        for k, v in json.load(open(occur_file)).items():
            self._occur_dict[k] = set(v)

    def retrieve(self, kw1:str, kw2:str):
        co_occur_index = self._occur_dict[kw1] & self._occur_dict[kw2]
        return [self._sents[idx] for idx in co_occur_index]


def count_line(file:str):
    with open(file) as f_in:
        return len(f_in.read().split('\n'))