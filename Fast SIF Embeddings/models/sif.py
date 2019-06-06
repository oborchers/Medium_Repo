import numpy as np
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.keyedvectors import BaseKeyedVectors

from sklearn.decomposition import TruncatedSVD

try:
# Import cython functions  
    from models.sif_inner import sif_embeddings_blas as sif_embedding
    CY_ROUTINES = 1
except ImportError as e:
    from models.sif_variants import sif_embeddings_4 as sif_embedding
    CY_ROUTINES = 0

# Define data types for use in cython
REAL = np.float32 
INT = np.intc

class SIF():

    def __init__(self, alpha=1e-3, components=1):
        """Class for computing the SIF embedding

        Parameters
        ----------
        alpha : float, optional
            Parameter which is used to weigh each individual word based on its probability p(w).
            If alpha = 1 train simply computes the average representation
        components : int, optional
            Number of principal components to remove from the sentence embedding

        Returns
        -------
        numpy.ndarray 
            SIF sentence embedding matrix of dim len(sentences) * dimension
        """
        self.alpha = float(alpha)
        self.components = int(components)


    def compute_principal_component(self, X,npc=1):
        """
        Source: https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: component_[i,:] is the i-th pc
        """
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0, algorithm="randomized")
        svd.fit(X)
        return svd.components_

    def remove_principal_component(self, X, npc=1):
        """
        Source: https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        Remove the projection on the principal components
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: XX[i, :] is the data point after removing its projection
        """
        pc = self.compute_principal_component(X, npc)
        if npc==1:
            XX = X - X.dot(pc.transpose()) * pc
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)
        return XX


    def precompute_sif_weights(self, wv, alpha=1e-3):
        """ Precompute the SIF weights

        Parameters
        ----------
        wv : `~gensim.models.keyedvectors.BaseKeyedVectors`
            A gensim keyedvectors child that contains the word vectors and the vocabulary
        alpha : float
            Parameter which is used to weigh each individual word based on its probability p(w).

        """
        if alpha > 0:
            corpus_size = 0
            wv.sif = np.zeros(shape=len(wv.vocab), dtype=REAL) 

            for k in wv.index2word:
                corpus_size += wv.vocab[k].count

            for idx, k in enumerate(wv.index2word):
                pw = wv.vocab[k].count / corpus_size
                wv.sif[idx] = alpha / (alpha+pw)
        else:
            wv.sif = np.ones(shape=len(wv.vocab), dtype=REAL)

    def precompute_sif_vectors(self, wv):
        """ Precompute the SIF Vectors

        Parameters
        ----------
        wv : `~gensim.models.keyedvectors.BaseKeyedVectors`
            A gensim keyedvectors child that contains the word vectors and the vocabulary
        """

        if not hasattr(wv, 'sif'):
            self.precompute_sif_weights(wv, self.alpha)
        if not hasattr(wv, 'sif_vectors'):
            wv.sif_vectors = (wv.vectors * wv.sif[:, None]).astype(REAL)

    def train(self, model, sentences, clear=True):
        """ Precompute the SIF Vectors

        Parameters
        ----------
        sentences : iterable
            An iterable which contains the sentences

        Returns
        -------
        numpy.ndarray 
            SIF sentence embedding matrix of dim len(sentences) * dimension
        """        
        if isinstance(model, BaseWordEmbeddingsModel):
            m = model.wv
        elif isinstance(model, BaseKeyedVectors):
            m = model
        else:
            raise RuntimeError("Model must be child of BaseWordEmbeddingsModel")

        self.precompute_sif_vectors(m)
        output = sif_embedding(sentences, m)

        if self.components > 0:
            output = self.remove_principal_component(output, self.components)

        if clear:
            m.sif_vectors = None
            m.sif = None

        return output