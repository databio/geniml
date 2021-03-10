
from collections import Counter
import csv
from gensim.models import Word2Vec
import gzip
import io
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
from numba import config, threading_layer
import numpy as np
import os
import pathlib
import pandas as pd
import re
import scipy.io
import seaborn as sns
from six.moves import cPickle as pickle #for performance
import tempfile
import umap

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = 'threadsafe'

class singlecellEmbedding(object):

    def embedding_avg(self, model, document):
        listOfWVs= []
        if type(document) is list: 
            for word in document:
                if word in model.wv.vocab:
                    listOfWVs.append(model[word])
            if(len(listOfWVs) == 0):
                return np.zeros([len(model[list(model.wv.vocab.keys())[0]])])
        else:
            for word in document.split(' '):
                if word in model.wv.vocab:
                    listOfWVs.append(model[word])
            if(len(listOfWVs) == 0):
                return np.zeros([len(model[list(model.wv.vocab.keys())[0]])])
        return np.mean(listOfWVs, axis=0)


    def document_embedding_avg(self, document_Embedding, model):
        document_Embedding_avg = {}
        for file, doc  in document_Embedding.items():
            document_Embedding_avg[file] = self.embedding_avg(model, doc)
        return document_Embedding_avg


    # shuffle the document to generate data for word2vec
    def shuffling(self, documents, shuffle_repeat):
        common_text = list(documents.values())
        training_samples = []
        training_samples.extend(common_text)
        for rn in range(shuffle_repeat):
            [(np.random.shuffle(l)) for l in common_text]
            training_samples.extend(common_text)
        return training_samples


    def trainWord2Vec(self, documents, window_size = 100,
                      dim = 100, min_count = 10, nothreads = 1):
        """
        Train word2vec algorithm
        """
        model = Word2Vec(sentences=documents, window=window_size,
                         size=dim, min_count=min_count, workers=nothreads)
        return model


    def label_split(self, s):
        return re.split(r'(^[^\d]+)', s)[1:]
        #return filter(None, re.split(r'(\d+)', s))


    # preprocess the labels
    def label_preprocessing(self, y):
        y_cell = []
        for y1 in y:
            y_cell.append(y1.split('_')[0])
        return y_cell


    # This function reduce the dimension using umap and plot 
    def UMAP_plot(self, data_X, y, title, nn, filename, umet,
                  plottitle, output_folder):
        np.random.seed(42)
        dp = 300
        # TODO: make points vector graphics too
        ump = umap.UMAP(a=None, angular_rp_forest=False, b=None,
                        force_approximation_algorithm=False, init='spectral',
                        learning_rate=1.0, local_connectivity=1.0,
                        low_memory=False, metric=umet, metric_kwds=None,
                        min_dist=0.1, n_components=2, n_epochs=1000,
                        n_neighbors=nn, negative_sample_rate=5,
                        output_metric=umet, output_metric_kwds=None,
                        random_state=42, repulsion_strength=1.0,
                        set_op_mix_ratio=1.0, spread=1.0,
                        target_metric='categorical', target_metric_kwds=None,
                        target_n_neighbors=-1, target_weight=0.5,
                        transform_queue_size=4.0, transform_seed=42,
                        unique=False, verbose=False)
        ump.fit(data_X) 
        ump_data = pd.DataFrame(ump.transform(data_X)) 
        print("Threading layer chosen: %s" % threading_layer())
        ump_data = pd.DataFrame({'UMAP 1':ump_data[0],
                                'UMAP 2':ump_data[1],
                                title:y})
        fig, ax = plt.subplots(figsize=(30,25))
        plate =(sns.color_palette("husl", n_colors=len(set(y))))
        sns.scatterplot(x="UMAP 1", y="UMAP 2", hue=title, s= 200,ax= ax,
                        palette = plate, sizes=(100, 900),
                        data=ump_data, #.sort_values(by = title),
                        rasterized=True)
        # TODO: only label a subset of the samples...
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper right", fontsize =  10,
                   markerscale=3, edgecolor = 'black')
        return fig


    def chunkify(self, csr, barcodes, n):
        """Yield successive n-sized chunks."""
        for i in range(0, csr.shape[1], n):
            yield csr[:,i:i + n], barcodes[i:i + n]


    def convertMM2doc2FileMap(self, arg):
        csr_slice, barcodes, temp_dir = arg
        documents = {}
        temp_file=tempfile.NamedTemporaryFile(dir=temp_dir.name, delete=False)
        for i in range(0, csr_slice.shape[1]):
            try:
                if(barcodes[i] not in documents):
                    documents[barcodes[i]] = []
                documents[barcodes[i]].extend(np.array([x + 1 for x in csr_slice[:,i].nonzero()[0].tolist()],dtype=str).tolist()) 
            except IndexError:
                #print("i: {}".format(i))  # DEBUG
                pass
        self.save_dict(documents, temp_file.name)
        documents = {}


    def buildDoc(self, docs, temp_dir, init=True):
        for f in os.listdir(temp_dir.name):
            d_file = os.path.join(os.path.dirname(os.path.abspath(temp_dir.name)), temp_dir.name, f)
            #print("file: {}".format(d_file))  # DEBUG
            doc = self.load_dict(d_file)
            if init:
                init = False
                docs = self.load_dict(d_file)
            else:
                self.mergeDict( docs, self.load_dict(d_file) )    
        return(docs)


    def save_dict(self, di_, filename_):
        with open(filename_, 'wb') as f:
            pickle.dump(di_, f)


    def load_dict(self, filename_):
        with open(filename_, 'rb') as f:
            try:
                ret_di = pickle.load(f)
                return ret_di
            except EOFError:
                print("Size (In bytes) of '%s':" %filename_, os.path.getsize(filename_))


    def mergeDict(self, d1, d2):
        # See: https://stackoverflow.com/questions/26910708/merging-dictionary-value-lists-in-python
        for key, value in d1.items():
            if key in d2:
                if type(value) is dict:
                    self.mergeDict(d1[key], d2[key])
                else:
                    if type(value) in (int, float, str):
                        d1[key] = [value]
                    if type(d2[key]) is list:
                        d1[key].extend(d2[key])
                    else:
                        d1[key].append(d2[key])
        for key, value in d2.items():
            if key not in d1:
                d1[key] = value



################################################################################
    def main(self, path_file, names_file, out_dir, title, nocells, noreads, 
             docs_file = None, w2v_model = None, shuffle_repeat = 5, 
             window_size = 100, dimension = 100,  min_count = 10, threads = 1,
             umap_nneighbours = 100, umap_metric = 'euclidean'):
        
        # Create pool *before* loading any data
        pool = mp.Pool(int(threads))

        print('-- Loading data... --')
        # TODO: read the mm file in chunks? Or load and write chunks, then
        #       split chunks across processors
        data = scipy.io.mmread(path_file)
        print('-- MatrixMarket file loaded --')
        data = data.tocsr()
        data = data[data.getnnz(1)>int(noreads)][:,data.getnnz(0)>int(nocells)]
        # TODO: test using the features file and whether that changes outcome and timing
        # features_filename = os.path.join(pathlib.Path(path_file).parents[0],
            # pathlib.Path(pathlib.Path(path_file).stem).stem + "_coords.tsv.gz")
        # barcodes_filename = os.path.join(pathlib.Path(path_file).parents[0],
            # pathlib.Path(pathlib.Path(path_file).stem).stem + "_names.tsv.gz")
        # feature_chr = [row[0] for row in csv.reader(gzip.open(features_filename, mode="rt"), delimiter="\t")]
        # feature_start = [row[1] for row in csv.reader(gzip.open(features_filename, mode="rt"), delimiter="\t")]
        # feature_end = [row[2] for row in csv.reader(gzip.open(features_filename, mode="rt"), delimiter="\t")]
        # features = [i + "_" + j + "_" + k for i, j, k in zip(feature_chr, feature_start, feature_end)] 
        # try:
            # features.remove('chr_start_end')
        # except ValueError:
            # pass
        # features = pd.DataFrame(features, columns=['region'])
        # print('-- features file loaded --')
        if names_file.lower().endswith('.gz'):
            barcodes = [row[0] for row in csv.reader(gzip.open(names_file, mode="rt"), delimiter="\t")]
        else:
            barcodes = [row[0] for row in csv.reader(gnames_file, delimiter="\t")]
        print('-- Sample names file loaded --')

        if docs_file:
            documents = load_dict(docs_file)
        else:
            docs_filename = os.path.join(out_dir, title + "_documents.pkl")
            documents = {}
            #print("out_dir: {}".format(out_dir))  # DEBUG
            temp_dir = tempfile.TemporaryDirectory(dir=out_dir) 
            #print("temp_dir: {}".format(temp_dir.name))  # DEBUG
            max_pos = data.shape[1]
            n = int(max_pos/mp.cpu_count())
            n = 1 if n < 1 else n # Don't allow value below 1
            n = 100 if n > 100 else n  # TODO: shrink this based on file size...
            args = ((chunk, names, temp_dir) for chunk, names in self.chunkify(data, barcodes, n))
            pool.map_async(self.convertMM2doc2FileMap, args)
            #clean up
            pool.close()
            pool.join()
            #print("convertDocPool2File complete")  # DEBUG
            documents = self.buildDoc(documents, temp_dir)
            print('-- Documents created --')
            temp_dir.cleanup()

        if not w2v_model:
            model_name = '_nocells{}_noreads{}_dim{}_win{}_mincount{}_shuffle{}.model'.format(
                str(nocells), str(noreads), str(dimension), str(window_size),
                str(min_count), str(shuffle_repeat))
            model_filename = os.path.join(out_dir, title + model_name)
            shuffeled_documents = self.shuffling(documents,
                                                 int(shuffle_repeat))
            model = self.trainWord2Vec(shuffeled_documents,
                                       window_size = int(window_size),
                                       dim = int(dimension),
                                       min_count = int(min_count),
                                       nothreads = int(threads))
            model.save(model_filename)
            print('-- Model created --')
        else:
            model = Word2Vec.load(w2v_model)

        print('Number of words in w2v model: ', len(model.wv.vocab))
        embeddings = self.document_embedding_avg(documents, model)
        X = pd.DataFrame(embeddings).values
        y = list(embeddings.keys())
        y = self.label_preprocessing(y)

        plot_name = '{}_nocells{}_noreads{}_dim{}_win{}_mincount{}_shuffle{}_umap_nneighbours{}_umap-metric{}.svg'.format(
            title, str(nocells), str(noreads), str(dimension), str(window_size),
            str(min_count), str(shuffle_repeat), str(umap_nneighbours),
            str(umap_metric))
        plot_filename = os.path.join(out_dir, plot_name)
        fig = self.UMAP_plot(X.T, y, 'single-cell', int(umap_nneighbours),
                             'Single-cell', umap_metric, 'RegionSet2vec', './')
        print('-- Saving UMAP plot... --')                     
        fig.savefig(plot_filename, format = 'svg')
        print('-- Pipeline Complete! --') 
