import os
import numpy as np
import random
import glob
import time
import datetime
import argparse
import os
from gitk.region2vec import utils

class BEDDataset:
    def __init__(self, args, file_list):
        self.links = []
        self.args = args
        self.meta_data = dict()
        self.file2idx = dict()
        random.seed(0)
        np.random.seed(0)
        with open(file_list, 'r') as f:        
            for idx,line in enumerate(f):
                filename = line.strip()
                self.links.append(filename)
                self.file2idx[filename] = idx
                
        self.nfiles = len(self.links)
  

    def regions2sentences_sampling(self, src_path, dst_path):
        for fname in self.links:
            # print(fname,flush=True)
            src_fname = os.path.join(src_path, fname)
            sentence = []
            probs = []
            with open(src_fname, 'r') as f:
                for line in f:
                    elements = line.strip().split('\t')
                    word = elements[0].strip()
                    sentence.append(word)
                    probs.append(float(elements[-2].strip()))
            probs = np.array(probs)
            probs = probs/probs.sum()
            sentence = np.array(sentence)
            
            sampled_sentence = np.random.choice(sentence, len(probs), p=probs)
            # sampled_sentence = list(set(sampled_sentence))
            sampled_sentence = sampled_sentence.tolist()
            str_sent = ' '.join(sampled_sentence)
            dst_fname = os.path.join(dst_path,fname)
            with open(dst_fname, 'w') as f:
                f.write(str_sent)
                f.write('\n')

    def regions2sentences(self, src_path, dst_path):
        for fname in self.links:
            src_fname = os.path.join(src_path, fname)
            sentence = []
            with open(src_fname, 'r') as f:
                for line in f:
                    elements = line.strip().split('\t')[0:3]
                    chr_name = elements[0].strip()
                    start = elements[1].strip()
                    end = elements[2].strip()
                    word = chr_name+':'+start+'-'+end
                    sentence.append(word)
            random.shuffle(sentence) #shuffle the regions in the sentence
            str_sent = ' '.join(sentence)
            dst_fname = os.path.join(dst_path,fname)
            with open(dst_fname, 'w') as f:
                f.write(str_sent)
                f.write('\n')

def main(args):
    
    DATA_FOLDER = os.path.join(args.save_dir, 'shuffled_datasets')
    src_path = args.tokenization_folder
    worker_id = args.worker_id
    random.seed(worker_id)
    dataset = BEDDataset(args, args.file_list)
    pool = args.pool
    utils.log('[{}] Creating shuffled datasets in \033[93m{}\033[00m (at most {} datasets coexist)'.format(worker_id,DATA_FOLDER, pool))
    
    for i in range(pool):
        name_used = os.path.join(DATA_FOLDER, f'pool{worker_id}-{i}used')
        name_using = os.path.join(DATA_FOLDER, f'pool{worker_id}-{i}using')
        name_creating = os.path.join(DATA_FOLDER, f'pool{worker_id}-{i}creating')
        name = os.path.join(DATA_FOLDER, f'pool{worker_id}-{i}')
        if os.path.exists(name_using):
            print('Folder exists')
            return
        if os.path.exists(name_used):
            print('Folder exists')
            return
        if os.path.exists(name):
            print('Folder exists')
            return
        if os.path.exists(name_creating):
            print('Folder exists')
            return
        os.makedirs(name_used)

    num_created = 0
    while True:
        if num_created == args.number:
            break
        #determine whether to create a new dataset
        folders = glob.glob(os.path.join(DATA_FOLDER, f'pool{worker_id}*used'))
        if len(folders) == 0:
            time.sleep(1) #wait for 10 seconds
            # print('Waiting for the data to be consumed',end="\r")
        else:
            #delete the used dataset and generate a new dataset in the same foler
            folder = folders[random.randint(0,len(folders)-1)]
            fname = folder.split('/')[-1][:-4]
            # print('[',datetime.datetime.now(),']','Find used dataset {}'.format(fname))
            os.system('rm -rf {}'.format(folder)) #delete the dataset
            os.makedirs(os.path.join(DATA_FOLDER, fname+'creating'))
            dpath = os.path.join(DATA_FOLDER, fname+'creating')
            if args.tokenization_mode == 'hard':
                dataset.regions2sentences(src_path, dpath)
            else:
                dataset.regions2sentences_sampling(src_path, dpath)

            num_created += 1
            # print('[',datetime.datetime.now(),']',' Created %dth dataset' % num_created)
            dst_name = os.path.join(DATA_FOLDER, fname)
            os.rename(dpath, dst_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Sentence Generation')
    parser.add_argument('--file_list', help='path to a file list')
    parser.add_argument('--tokenization_mode', help='tokenization mode')
    parser.add_argument('--tokenization_folder', help='path to the tokenized regions')
    parser.add_argument('--save_dir', help='parent folder to generated shuffled datasets')
    parser.add_argument('--pool', type=int, default=3, help='maximum number of shuffled datasets before consuming one')
    parser.add_argument('--worker_id', type=int, default=0, help='maximum number of shuffled datasets before consuming one')
    parser.add_argument('--number', type=int, default=1000, help='total number of shuffled datasets')

    
    args = parser.parse_args()

    main(args)

    


    



