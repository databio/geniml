import pandas as pd
import multiprocessing as mp
from gensim.models import Word2Vec
from scipy.io import mmread
from numba import config, njit, threading_layer
import numpy as np
import matplotlib
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import datetime
import re

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = 'threadsafe'

class singlecellEmbedding(object):
    
    # Preprocessing to filter based on the number of cells and reads
    def preprocessing(self, data, nocells, noreads):
        data.drop(data.loc[data[list(data)[3:-1]].sum(axis=1)< nocells].index, inplace=True)

        data.drop(columns=data[list(data)[3:-1]].columns[data[list(data)[3:-1]].sum()< noreads], inplace=True)

        data['region'] = data['chr'] + '_' + data['start'].apply(str) + '_' + data['end'].apply(str)

        data = data[list(data)[3:]]
        print(data.shape)
        return data


    def embedding_avg(self, model, document):
        listOfWVs= []
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


    # preprocess the data for word2vec 
    def convertMat2document(self, data):
        documents = {}
        for cell in list(data)[:-1]:
            index = [index for index, value in enumerate(data[cell]) if value >= 1]
            doc = ' '.join(data.iloc[index]['region'])
            documents[cell] = doc
        return documents



    # shuffle the document to generate data for word2vec
    def shuffling(self, document_universe, shuffle_repeat):
        common_text = [value.split(' ')  for key, value in document_universe.items()]
        training_samples = []
        training_samples.extend(common_text)

        for rn in range(shuffle_repeat):
            [(np.random.shuffle(l)) for l in common_text]
            training_samples.extend(common_text)
        return training_samples



    def trainWord2vec(self, documents, window_size = 100,
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
            #y_cell.append('-'.join(y1.replace('singles-', '').replace('BM1077-', '').split('-')[0:-1])) # GSE749412
            y_cell.append(re.split(r'(^[^\d]+)', y1)[1:][0]) # Alexandre and 10X
        return y_cell



    # This function reduce the dimension using umap and plot 
    def UMAP_plot(self, data_X, y, title, nn, filename,
                  plottitle, output_folder):

        np.random.seed(42)
        dp = 300

        ump = umap.UMAP(a=None, angular_rp_forest=False, b=None,
                        force_approximation_algorithm=False, init='spectral',
                        learning_rate=1.0, local_connectivity=1.0,
                        low_memory=False, metric='euclidean', metric_kwds=None,
                        min_dist=0.1, n_components=2, n_epochs=1000,
                        n_neighbors=nn, negative_sample_rate=5,
                        output_metric='euclidean', output_metric_kwds=None,
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


    def process_frame(self, data, nocells, noreads):
            # process data frame
            data = self.preprocessing(data, int(nocells), int(noreads))
            return self.convertMat2document(data)

    
    def main(self, path_file, nocells, noreads, w2v_model, mm_format = False, 
             shuffle_repeat = 1, window_size = 100, dimension = 100, 
             min_count = 10, threads = 1, chunks = 10, umap_nneighbours = 96,
             model_filename = './model.model', plot_filename = './name.jpg'):

        # TODO: use SciPy to load a MatrixMarket format and convert to dense format file
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.todense.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html
        
        # TODO: if path_file outdir doesn't exist, create it!
        
        if mm_format:
            print('Loading data via mmread()')
            mm_file = mmread(path_file)
            data = mm_file.todense()
        else:
            #print('Loading data via pandas.read_csv()')  # DEBUG

            col_names = pd.read_csv(path_file, nrows=0, sep="\t").columns
            types_dict = {'chr': str, 'start': int, 'end': int}
            types_dict.update({col: 'int8' for col in col_names if col not in types_dict})

            reader = pd.read_csv(path_file, sep="\t",
                                 chunksize=chunks, dtype=types_dict,
                                 keep_default_na=False, error_bad_lines=False)
            pool = mp.Pool(int(threads))
            
            funclist = []
            for df in reader:
                # process each data frame
                f = pool.apply_async(self.process_frame,[df, nocells, noreads])
                funclist.append(f)

            chunk_no = 0
            documents = {}
            for f in funclist:
                chunk_no += 1
                if chunk_no is 1:
                    #print('Loaded first chunk')  # DEBUG
                    documents = f.get()
                else:
                    #print('Loading {}th chunk'.format(str(chunk_no)))  # DEBUG
                    tmp = f.get()
                    documents = {key: documents[key] + " " + tmp[key] for key in documents}

            # data = pd.read_csv(path_file, sep='\t', lineterminator='\n',
                               # dtype=types_dict, keep_default_na=False,
                               # error_bad_lines=False)
            #data.columns = data.columns.str.strip().str.lower()
        #print('number of peaks: ', len(data))
        #data = self.preprocessing(data, int(nocells), int(noreads))
        #print('number of peaks after filtering: ', len(data))
        
        # TODO: load in chunks, and generate documents, then concatenate the documents
        #documents = self.convertMat2document(data)
        
        #data_frames = np.array_split(data, 2)
        #data1 = data_frames[0]
        #data2 = data_frames[1]
        
        #doc1 = self.convertMat2document(data1)
        #doc2 = self.convertMat2document(data2)
        
        # Recombine documents into single document
        #documents2 = {key: doc1[key] + " " + doc2[key] for key in doc1}
        
        print('number of documents: ', len(documents))

        if not w2v_model:
            shuffeled_documents = self.shuffling(documents, int(shuffle_repeat))
            print('number of shuffled documents: ', len(shuffeled_documents))
            print('Dimension: ', dimension)
            model = self.trainWord2vec(shuffeled_documents,
                                       window_size = int(window_size),
                                       dim = int(dimension),
                                       min_count = int(min_count),
                                       nothreads = int(threads))
            model.save(model_filename)
        else:
            model = Word2Vec.load(w2v_model)

        print('Number of words in w2v model: ', len(model.wv.vocab))
        document_Embedding_avg = self.document_embedding_avg(documents, model)
        X = pd.DataFrame(document_Embedding_avg).values
        y = list(document_Embedding_avg.keys())
        y = self.label_preprocessing(y)

        #print(Counter(y))
        fig = self.UMAP_plot(X.T, y, 'single-cell',
                             int(umap_nneighbours), 'Single-cell',
                             'RegionSet2vec', './')
        print('Saving UMAP plot...')                     
        fig.savefig(plot_filename, format = 'svg')
        print('DONE!') 