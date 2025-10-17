import numpy as np
cimport numpy as cnp

from .tree cimport Tree, free_tree, predict_proba_tree, predict_labels_tree
from .evolution cimport find_best_tree

cdef class EvolutionaryTree:
    cdef Tree* tree

    def __cinit__(self):
        self.tree = NULL

    def fit(self, cnp.ndarray[cnp.float32_t, ndim=2] X, cnp.ndarray[cnp.int32_t, ndim=1] y,
            int pop_size=100,
            int tournament_size=3,
            int n_elites=1,
            float crossover_rate=0.9,
            float mutation_rate=0.1,
            int max_depth=10,
            int max_generations=50,
            int patience=5,
            float tol=1e-3,
            int random_state=-1
    ):
        if self.tree is not NULL:
            free_tree(self.tree)
        self.tree = find_best_tree(
            X, y,
            pop_size=pop_size,
            tournament_size=tournament_size,
            n_elites=n_elites,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            max_depth=max_depth,
            max_generations=max_generations,
            patience=patience,
            tol=tol,
            random_state=random_state
        )

    def predict_proba(self, cnp.ndarray[cnp.float32_t, ndim=2] X):
        if self.tree is NULL:
            raise ValueError("The model has not been fitted yet.")
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[cnp.float32_t, ndim=1] probabilities = np.empty(n_samples, dtype=np.float32)
        cdef float[:] probabilities_view = probabilities
        predict_proba_tree(self.tree, X, probabilities_view, n_samples)
        return probabilities

    def predict(self, cnp.ndarray[cnp.float32_t, ndim=2] X):
        if self.tree is NULL:
            raise ValueError("The model has not been fitted yet.")
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
        cdef float[:] predictions_view = predictions
        predict_labels_tree(self.tree, X, predictions_view, n_samples)
        return predictions

    def __dealloc__(self):
        if self.tree is not NULL:
            free_tree(self.tree)
