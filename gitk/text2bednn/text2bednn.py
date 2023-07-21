from gitk.models import ExModel
from ..search import EmSearchBackend

class TextToBedNN(ExModel):
    """ A Extended Model for TextToBed using a FFNN. """
    def __init__(self, model, universe, tokenizer):
        super().__init__(model, universe, tokenizer)

    def embed(self, region_set: RegionSet) -> np.ndarray:
        """ Embed a region set using the model """



class TextToBedNNSearchInterface(object):
    def __init__(self, exmodel: TextToBedNN, region_set_backend: EmSearchBackend):
        self.exmodel = exmodel
        self.region_set_backend = region_set_backend

    def nlsearch(self, query: str, k: int = 10) -> embeddings:
        """Given an input natural lange, suggest region sets"""

        # first, get the embedding of the query string
        query_embedding = self.tum.embed(query)
        # then, use the query embedding to predict the region set embedding
        region_set_embedding = self.tum.str_to_region_set(query_embedding)
        # finally, use the region set embedding to search for similar region sets
        return self.region_set_backend.search(region_set_embedding, k)



# Example of how to use the TextToBedNN Search Interface

betum = BEDEmbedTUM(RSC, universe, tokenizer)
embeddings = betum.compute_embeddings()
T2BNNSI = TextToBedNNSearchInterface(betum, embeddings)  # ???
