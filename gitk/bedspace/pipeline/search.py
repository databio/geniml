import logging

from ..const import DEFAULT_NUM_SEARCH_RESULTS, PKG_NAME

_LOGGER = logging.getLogger(PKG_NAME)


def run_scenario1(
    query: str,
    distances: str,
    num_results: int = DEFAULT_NUM_SEARCH_RESULTS,
    output: str,
):
    """
    Run the search command. This is for scenario 1: Give me a label, I'll return region sets.

    :param query: The query string (a label).
    :param distances: The path to the distances file.
    :param num_results: The number of results to return.
    :param output: The path to save the barplots.
    """
    _LOGGER.info("Running search...")

    # PLACE SEARCH CODE HERE
    searchterm = query
    
    distance = pd.read_csv(distances)
    distance.file_label = distance.file_label.str.lower()
    distance.search_term = distance.search_term.str.lower()
    distance = distance.drop_duplicates()
    search_table = pd.pivot_table(distance, values='score', index=['filename'],columns=['search_term']).reset_index()
    search_table = pd.merge(distance[['filename','file_label']].drop_duplicates(), search_table, on = 'filename').drop_duplicates()
    search_table= search_table.merge(meta_test, left_on='filename', right_on='file_name')

    if(search=='cell'):
        ind=0
        training_labels = cell_types
    else:
        ind=1
        training_labels=targets

    training_labels=sorted(training_labels)
    
    ### ???
    training_labels = list(search_table)[2:-4]

    len_targets = len(search_table.file_label[0].split(' _'))-1

    search_table.file_label = search_table.file_label.str.split(' _', expand=True)[np.min([ind, len_targets])]
    

    
    search_table = search_table[['filename','file_label', 'original_label']+(training_labels)]
    search_table['predicted_label'] = search_table[list(search_table)[3:]].idxmin(axis=1)
    


    nof=len(search_table[search_table.file_label.str.contains(searchterm)])
    df=search_table[['filename', 'file_label', 'original_label', searchterm]].sort_values(by=[searchterm])[0:num_results]
    df=df.sort_values(by=[searchterm], ascending=False)

    df['color']='gray'
    df.loc[df.file_label.str.contains(searchterm), 'color'] = 'green'
    if(len(df[df.color=='green']) == nof):
        df.loc[(df.color!='green'), 'color'] = 'gray'

    df[searchterm] = 1 - df[searchterm]

    plt= df.plot.barh(x='original_label', y=searchterm, figsize=(10,7), fontsize=16, color=list(df['color']))
    plt.set_xlabel('Similarity', fontsize=15)
    plt.set_ylabel('original_label', fontsize=15)

    plt.axis(xmin=0.5, xmax=1.01)

    plt.figure.savefig('{}/figures/S1/{}_nof{}.svg'.format(output, searchterm, nof), format = 'svg', bbox_inches='tight')
        

        

def run_scenario2(
    query: str,
    distances: str,
    num_results: int = DEFAULT_NUM_SEARCH_RESULTS,
    output: str,
):
    """
    Run the search command. This is for scenario 2: Give me a region set, I'll return labels.

    :param query: The query string (a path to a file).
    :param distances: The path to the distances file.
    :param num_results: The number of results to return.
    :param output: The path to save the barplots.
    """

    _LOGGER.info("Running search...")

    file = query
    
    # PLACE SEARCH CODE HERE
    distance =pd.read_csv(distances)
    distance.file_label = distance.file_label.str.lower()
    distance.search_term = distance.search_term.str.lower()
    distance = distance.drop_duplicates()
    search_table = pd.pivot_table(distance, values='score', index=['filename'],columns=['search_term']).reset_index()
    search_table= search_table.merge(meta_test, left_on='filename', right_on='file_name')
    search_table = pd.merge(distance[['filename','file_label']].drop_duplicates(), search_table, on = 'filename').drop_duplicates()

    if('cell' in search):
        ind=0
        training_labels=cell_types
    else:
        ind=1
        training_labels=targets

    training_labels=sorted(training_labels)


    len_targets = len(search_table.file_label[0].split(' _'))-1

    search_table.file_label = search_table.file_label.str.split(' _', expand=True)[np.min([ind, len_targets])]
    
          
        
    search_table = search_table[['filename','file_label', 'original_label']+(training_labels)]
    search_table['predicted_label'] = search_table[list(search_table)[3:]].idxmin(axis=1)

    i=0
    b=search_table
    all_weights=[]
    for fil in b.filename:
        c=b[b.filename==fil]['file_label']
        i=c.index[0]

        a = b[b.filename==fil][list(b[b.filename==fil])[3:-1]]
        weights=[]

        ol=b[b.filename==fil]['file_label'][i]
        for lb in list(a):
            if((a[lb][i]) == 0):
                weights.append((fil, ol, lb,  (1-a[lb][i])))
            else:
                weights.append((fil, ol, lb, (1-a[lb][i])))


        all_weights.extend(weights)

    
        i+=1


    X = pd.DataFrame(all_weights).rename(columns={0:'Filename', 1:'Filelabel', 2: 'AllLabels', 3:'Distance_score'})


#     for file in list(set(X.Filename)):
    df = X[X.Filename == file].sort_values(by=['Distance_score'], ascending = False)[0:10]
    df= df.sort_values(by=['Distance_score'], ascending = True)
    df['color']='green'
    plt= df.plot.barh(x='AllLabels', y='Distance_score', figsize=(8,5), fontsize=16, color=list(df['color']))
    plt.set_xticks(np.arange(0.5,1.1, 0.1))
    plt.set_xlabel('Similarity', fontsize=15)
    plt.set_ylabel('original_label', fontsize=15)

    plt.figure.savefig('../outputs/bedembed_output/figures/S2/'+file.split('/')[-1]+'.svg', format = 'svg', bbox_inches='tight', dpi =300)

    
    
def run_scenario3(
    query: str,
    distances: str,
    num_results: int = DEFAULT_NUM_SEARCH_RESULTS,
    output: str,
):
    """
    Run the search command. This is for scenario 3: Give me a region set, I'll return region sets.

    :param query: The query string (a path to a file).
    :param distances: The path to the distances file.
    :param num_results: The number of results to return.
    :param output: The path to save the barplots.
    
    """

    _LOGGER.info("Running search...")

    # PLACE SEARCH CODE HERE
    
    
