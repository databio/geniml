import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# Ensure pyqt5 is upgraded (pip install --user --upgrade pyqt5
from collections import Counter
import csv
from gensim.models import Word2Vec
import gzip
import io
import multiprocessing as mp
from numba import config, threading_layer
import numpy as np
import os
import pathlib
import pandas as pd
import re
import scipy.io
from six.moves import cPickle as pickle #for performance
import sys
import tempfile
import umap
from sklearn.preprocessing import normalize
import vaex
import log
from itertools import islice

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = 'threadsafe'

log = log.getLogger(__name__)

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
        # sg=0 Training algorithm: 1 for skip-gram; otherwise CBOW.
        model = Word2Vec(sentences=documents, window=window_size,
                         size=dim, min_count=min_count,
                         workers=nothreads)
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


    # This function reduces the dimension using umap and plot 
    def UMAP_plot(self, data_X, y, title, nn, filename, umet,
                  rasterize=False):
        np.random.seed(42)
        # TODO: make low_memory a tool argument
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
        log.info(f'-- {timestamp} Fitting UMAP data --\n')
        # Must `pip install --user --upgrade pynndescent` for large data
        ump.fit(data_X)
        ump_data = pd.DataFrame(ump.transform(data_X))
        ump_data = pd.DataFrame({'UMAP 1':ump_data[0],
                                'UMAP 2':ump_data[1],
                                title:y})
        ump_data.to_csv(filename, index=False)
        log.info(f'-- Saved UMAP data as {filename} --\n')
        fig, ax = plt.subplots(figsize=(8,6.4))
        plt.rc('font', size=11)
        plate =(sns.color_palette("husl", n_colors=len(set(y))))
        sns.scatterplot(x="UMAP 1", y="UMAP 2", hue=title, s= 10,ax= ax,
                        palette = plate, sizes=(10, 40),
                        data=ump_data, #.sort_values(by = title),
                        rasterized=rasterize)
        # TODO: only label a subset of the samples...
        plt.legend(bbox_to_anchor=(1.1,1), loc="upper right", fontsize =  11,
                   markerscale=2, edgecolor = 'black')
        return fig


    def chunkify(self, csr, barcodes, n):
        """Yield successive n-sized chunks."""
        for i in range(0, csr.shape[1], n):
            yield csr[:,i:i + n], barcodes[i:i + n]


    def chunkitionary(self, a_dict, n):
        """Yield successive n-sized chunks."""
        it = iter(a_dict)
        for i in range(0, len(a_dict), n):
            yield {k:a_dict[k] for k in islice(it, n)}


    def convertMM2doc(self, mtx_file):
        documents = {}
        with open(mtx_file) as src:
            for line in src:
                row, col, entry = line.strip().split()
                if col not in documents:
                    documents[str(col)] = []
                val = sys.intern(row)
                documents[col].append(val)
        return documents


    def chunkMMfile(self, mmfile, temp_dir):
        # REQUIRES unix sorted -k2,2n
        with gzip.open(mmfile, 'rb') as src:
            row, col, entry = int, int, int
            lineno = 0
            sample = None
            for line in src:
                lineno += 1
                if lineno <= 3:
                    # Skip the mtx header lines
                    pass
                else:
                    if not sample:
                        # Open first file
                        temp_file=tempfile.NamedTemporaryFile(dir=temp_dir.name, suffix='.mtx', delete=False)
                        chunk = open(temp_file.name, mode='w')
                    row, col, entry = line.decode('utf-8').strip().split()
                    if (sample) and (col != sample):
                        # Close the last file
                        chunk.close()
                        # Open a NEW file!
                        temp_file=tempfile.NamedTemporaryFile(dir=temp_dir.name, suffix='.mtx', delete=False)
                        chunk = open(temp_file.name, mode='w')
                        # Write the first value of this new sample
                        sample = col
                        out_line = ' '.join([row, col, entry])
                        _ = chunk.write(f'{out_line}\n')
                    else:
                        sample = col
                        out_line = ' '.join([row, col, entry])
                        _ = chunk.write(f'{out_line}\n')
            chunk.close()


    def chunkMMfile2doc(self, mmfile):
        # REQUIRES unix sorted -k2,2n
        documents = {}
        with gzip.open(mmfile, 'rb') as src:
            lineno = 0
            sample = None
            for line in src:
                row, col, entry = line.decode('utf-8').strip().split()
                if not sample:
                    # set initial key
                    documents[str(col)] = []
                if (sample) and (col != sample):
                    # set new key
                    documents[str(col)] = []
                    # Write the first value of this new sample
                    sample = col
                    val = sys.intern(row)
                    documents[col].append(val)
                else:
                    sample = col
                    val = sys.intern(row)
                    documents[col].append(val)
        return documents


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


    def convertMM2doc2FileMapInterned(self, arg):
        csr_slice, barcodes, temp_dir = arg
        documents = {}
        temp_file=tempfile.NamedTemporaryFile(dir=temp_dir.name, delete=False)
        for i in range(0, csr_slice.shape[1]):
            try:
                if(barcodes[i] not in documents):
                    documents[barcodes[i]] = []
                values = (np.array(
                    [x + 1 for x in csr_slice[:,i].nonzero()[0].
                        tolist()],dtype=str).tolist())
                interned = [sys.intern(val) for val in values]
                documents[barcodes[i]].extend(interned) 
            except IndexError:
                #print("i: {}".format(i))  # DEBUG
                pass
        self.save_dict(documents, temp_file.name)
        documents = {}


    def buildDoc(self, docs, temp_dir, init=True):
        for f in os.listdir(temp_dir.name):
            d_file = os.path.join(os.path.dirname(
                os.path.abspath(temp_dir.name)), temp_dir.name, f)
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


    def buildDict(self, mtx, SIZE=100_000):
        documents = {}
        for i1, i2, chunk in mtx.evaluate_iterator(mtx[:,0], chunk_size=SIZE):
            for x in chunk:
                row, col, entry = x.as_py().split()
                #print(f"{row}, {col}, {entry}")
                if col not in documents:
                    documents[str(col)] = []
                val = sys.intern(row)
                documents[col].append(val)
        return documents


    def replaceKeys(self, a_dict, new_keys):
        for key in list(a_dict.keys()):
            try:
                new_key = new_keys[int(key)-1][0]
                a_dict[new_key] = a_dict.pop(key)
            except (KeyError, AssertionError) as err:
                print(f"err: {err}")
                pass


    def replaceValues(self, a_dict, new_values):
        for key in list(a_dict.keys()):
            try:
                int_list = list(map(int, a_dict[key]))
                a_dict[key] = [sys.intern(new_values[i-1]) for i in int_list]
            except:
                e = sys.exc_info()[0]
                print(f"Exception: {e}")
                pass



################################################################################
    def main(self, path_file, names_file, coords_file, out_dir, nocells, noreads, 
             title = "scembed", docs_file = None, w2v_model = None,
             embed_file = None, shuffle_repeat = 5, window_size = 100,
             dimension = 100, min_count = 10, threads = 1,
             umap_nneighbours = 100, umap_metric = 'euclidean', 
             rasterize=False, alt_method=False, v2_method=False,
             interned=False, use_vaex=False, loglevel='DEBUG'):
        
        # Create pool *before* loading any data
        #pool = mp.Pool(int(threads))
        
        log_file = os.path.join(out_dir, title + "_log.txt")
        numeric_level = getattr(log, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {loglevel}")
        log.basicConfig(
            filename=log_file,
            filemode='a',
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p', encoding='utf-8',
            level=numeric_level
        )        

        log.info(f'loading data')

        # TODO: read the mm file in chunks? Or load and write chunks, then
        #       split chunks across processors
        if alt_method:
            if docs_file:
                documents = self.load_dict(docs_file)
            else:
                # Requires a unix sorted input file (-k2,2n -k1,1n)
                temp_dir = tempfile.TemporaryDirectory(dir=out_dir)
                self.chunkMMfile(path_file, temp_dir)
                log.info(f'MTX file split on samples')
                init = False
                documents = {} 
                for f in os.listdir(temp_dir.name):
                    mtx_file = os.path.join(os.path.dirname(
                            os.path.abspath(temp_dir.name)), temp_dir.name, f)
                    if init:
                        init = False
                        documents = self.convertMM2doc(mtx_file)
                    else:
                        tmp = self.convertMM2doc(mtx_file)
                        self.mergeDict( documents, tmp )
                log.info(f'len(documents.keys()): {len(documents.keys())}')
                #swap doc keys with barcodes by index
                if names_file.lower().endswith('.gz'):
                    pd_barcodes = pd.read_csv(
                        names_file, compression='gzip', header=None)
                else:
                    pd_barcodes = pd.read_csv(names_file,  header=None)
                for key in list(documents.keys()):
                    # TODO: use number of files in os.listdir as upper limit as well
                    try:
                        new_key = pd_barcodes.loc[int(key)-1,0]
                    except KeyError:
                        log.info(f'old key: {key}')
                        log.info(f'len(pd_barcodes): {len(pd_barcodes.index)}')
                        pass
                    documents[new_key] = documents.pop(key)
                docs_filename = os.path.join(out_dir, title + "_alt_documents.pkl")
                self.save_dict(documents, docs_filename)
                temp_dir.cleanup()
                log.info(f'Saved documents as {docs_filename}')
        elif v2_method:
            if docs_file:
                documents = self.load_dict(docs_file)
            else:
                # Requires a unix sorted input file (-k2,2n -k1,1n)
                documents = self.chunkMMfile2doc(path_file)
                #swap doc keys with barcodes by index
                if names_file.lower().endswith('.gz'):
                    pd_barcodes = pd.read_csv(
                        names_file, compression='gzip', header=None)
                else:
                    pd_barcodes = pd.read_csv(names_file,  header=None)
                for key in list(documents.keys()):
                    # TODO: use number of files in os.listdir as upper limit as well
                    try:
                        new_key = pd_barcodes.loc[int(key)-1,0]
                    except KeyError:
                        log.info(f'old key: {key}')
                        log.info(f'len(pd_barcodes): {len(pd_barcodes.index)}')
                        pass
                    documents[new_key] = documents.pop(key)
                docs_filename = os.path.join(out_dir, title + "_v2_documents.pkl")
                self.save_dict(documents, docs_filename)
                log.info(f'Saved documents as {docs_filename}')
        elif use_vaex:
            if docs_file:
                documents = self.load_dict(docs_file)
            else:
                if os.path.exists(path_file + ".hdf5"):
                    df = vaex.open(path_file + ".hdf5")
                else:
                    # initialize
                    df = vaex.from_csv(path_file, sep="\t", convert=True,
                                       chunk_size=5_000_000, copy_index=False,
                                       header=[0,1,2])

                documents = self.buildDict(df)

                if os.path.exists(names_file + ".hdf5"):
                    names = vaex.open(names_file + ".hdf5")
                else:
                    # initialize
                    names = vaex.from_csv(names_file, sep="\t", convert=True,
                                          chunk_size=5_000_000, copy_index=False,
                                          header=None)

                if os.path.exists(coords_file + ".hdf5"):
                    feats = vaex.open(coords_file + ".hdf5")
                else:
                    # initialize
                    feats = vaex.from_csv(coords_file, sep="\t", convert=True,
                                          chunk_size=5_000_000, copy_index=False)
                
                self.replaceKeys(documents, names)
                regions = feats['chr'] + " " + feats['start'].astype(str) + " " + feats['end'].astype(str)
                regions = regions.tolist()
                self.replaceValues(documents, regions)

                docs_filename = os.path.join(out_dir, title + "_vaex_documents.pkl")
                self.save_dict(documents, docs_filename)
                log.info(f'Saved documents as {docs_filename}')
        else:
            if names_file.lower().endswith('.gz'):
                barcodes = [row[0] for row in csv.reader(gzip.open(names_file, mode="rt"), delimiter="\t")]
            else:
                barcodes = [row[0] for row in csv.reader(names_file, delimiter="\t")]

            log.info(f'Sample names file loaded')

            if docs_file:
                documents = self.load_dict(docs_file)
            else:
                data = scipy.io.mmread(path_file)
                data = data.tocsr()
                # Scale input vectors individually to unit norm (vector length).
                data = normalize(data, norm='l1', axis=1, copy=False)
                log.info(f'MatrixMarket file loaded')
                data = data[data.getnnz(1)>int(noreads)][:,data.getnnz(0)>int(nocells)]
                docs_filename = os.path.join(out_dir, title + "_documents.pkl")
                documents = {}
                temp_dir = tempfile.TemporaryDirectory(dir=out_dir)
                max_pos = data.shape[1]
                n = int(max_pos/mp.cpu_count())
                n = 1 if n < 1 else n # Don't allow value below 1
                n = 100 if n > 100 else n  # TODO: shrink this based on file size...
                args = ((chunk, names, temp_dir) for chunk, names in self.chunkify(data, barcodes, n))
                if interned:
                    pool.map_async(self.convertMM2doc2FileMapInterned, args)
                else:
                    pool.map_async(self.convertMM2doc2FileMap, args)
                #clean up
                pool.close()
                pool.join()
                documents = self.buildDoc(documents, temp_dir)
                log.info(f'Documents created')
                self.save_dict(documents, docs_filename)
                temp_dir.cleanup()
                data = None

        if not w2v_model:
            model_name = '_nocells{}_noreads{}_dim{}_win{}_mincount{}_shuffle{}.model'.format(
                str(nocells), str(noreads), str(dimension), str(window_size),
                str(min_count), str(shuffle_repeat))
            model_filename = os.path.join(out_dir, title + model_name)
            log.info(f'Shuffling documents')
            shuffeled_documents = self.shuffling(documents,
                                                 int(shuffle_repeat))
            log.info(f'Constructing model')
            model = self.trainWord2Vec(shuffeled_documents,
                                       window_size = int(window_size),
                                       dim = int(dimension),
                                       min_count = int(min_count),
                                       nothreads = int(threads))
            model.save(model_filename)
            log.info(f'Model saved as: {model_filename}')
        else:
            model = Word2Vec.load(w2v_model)

        log.info(f'Number of words in w2v model: {len(model.wv.vocab)}')

        if not embed_file:
            embeddings = self.document_embedding_avg(documents, model)
            embeddings_dictfile = os.path.join(
                out_dir, title + "_embeddings.pkl")
            self.save_dict(embeddings, embeddings_dictfile)
            embeddings_csvfile = os.path.join(
                out_dir, title + "_embeddings.csv")
            (pd.DataFrame.from_dict(data=embeddings, orient='index').
             to_csv(embeddings_csvfile, header=False))
            log.info(f'Embeddings file saved as {embeddings_csvfile}')
        else:
            embeddings = self.load_dict(embed_file)

        X = pd.DataFrame(embeddings).values
        y = list(embeddings.keys())
        y = self.label_preprocessing(y)

        log.info(f'Generating plot')
        coordinates_csvfile = os.path.join(out_dir, title + "_xy_coords.csv")
        plot_name = '{}_nocells{}_noreads{}_dim{}_win{}_mincount{}_shuffle{}_umap_nneighbours{}_umap-metric{}.svg'.format(
            title, str(nocells), str(noreads), str(dimension), str(window_size),
            str(min_count), str(shuffle_repeat), str(umap_nneighbours),
            str(umap_metric))
        plot_filename = os.path.join(out_dir, "figs", plot_name)
        fig = self.UMAP_plot(X.T, y, 'single-cell', int(umap_nneighbours),
                             coordinates_csvfile, umap_metric, rasterize,
                             log_file)
        log.info(f'Saving UMAP plot')
        fig.savefig(plot_filename, format = 'svg')
        log.info(f'Pipeline Complete!')

