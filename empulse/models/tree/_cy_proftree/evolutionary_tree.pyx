import numpy as np
cimport numpy as cnp

from .tree cimport Tree, free_tree, predict_proba_tree, predict_labels_tree, serialize_tree, deserialize_tree
from .evolution cimport (
    evolve_forest_stochastic, evolve_forest_native, EvolutionResult, ExpectedMaxProfitOfGroups, NativeFitness
)

cdef class EvolutionaryTree:
    cdef Tree* tree
    cdef public int n_generations

    def __cinit__(self):
        self.tree = NULL
        self.n_generations = 0

    def fit_custom(
        self,
        cnp.ndarray[cnp.float32_t, ndim=2] X,
        cnp.ndarray[cnp.int32_t, ndim=1] y,
        int pop_size=100,
        float crossover_rate=0.2,
        float grow_rate=0.2,
        float prune_rate=0.2,
        float mutate_split_rate=0.2,
        float mutate_value_rate=0.2,
        int max_depth=10,
        int min_samples_split=2,
        int min_samples_leaf=1,
        float alpha=0.0,
        int max_generations=50,
        int patience=5,
        float tol=1e-3,
        object fitness_function=None,
        int random_state=-1,
        int n_jobs=1,
        bint fitness_from_leaves=False,
        bint cache_samples=True,
    ):
        if self.tree is not NULL:
            free_tree(self.tree)

        if fitness_function is None:
            from sklearn.metrics import accuracy_score
            fitness_function = accuracy_score

        # Samples are routed through the tree by pointer to their row, so rows must be contiguous.
        X = np.ascontiguousarray(X)

        cdef EvolutionResult result = evolve_forest_stochastic(
            X,
            y,
            fitness_function,
            pop_size,
            max_depth,
            max_generations,
            min_samples_split,
            min_samples_leaf,
            crossover_rate,
            grow_rate,
            prune_rate,
            mutate_split_rate,
            mutate_value_rate,
            patience,
            tol,
            alpha,
            random_state,
            n_jobs,
            fitness_from_leaves,
            cache_samples,
        )
        self.tree = result.tree
        self.n_generations = result.n_generations

    def fit_max_profit(
        self,
        cnp.ndarray[cnp.float32_t, ndim=2] X,
        cnp.ndarray[cnp.int32_t, ndim=1] y,
        float tp_benefit=0.0,
        float tn_benefit=0.0,
        float fp_cost=0.0,
        float fn_cost=0.0,
        int pop_size=100,
        float crossover_rate=0.2,
        float grow_rate=0.2,
        float prune_rate=0.2,
        float mutate_split_rate=0.2,
        float mutate_value_rate=0.2,
        int max_depth=10,
        int min_samples_split=2,
        int min_samples_leaf=1,
        float alpha=0.0,
        int max_generations=50,
        int patience=5,
        float tol=1e-3,
        int random_state=-1,
        int n_jobs=1,
        bint cache_samples=True,
        object closed_form=None,
        size_t expected_max_profit_of_groups=0,
    ):
        """
        Evolve trees whose fitness is computed natively: the maximum profit for the four class values,
        or, given ``closed_form`` and the address of the compiled ``expected_max_profit_of_groups``,
        the expected maximum profit that ``closed_form`` (a ``ClosedFormExpectedMaxProfit``) describes.
        """
        if self.tree is not NULL:
            free_tree(self.tree)

        # Samples are routed through the tree by pointer to their row, so rows must be contiguous.
        X = np.ascontiguousarray(X)

        cdef NativeFitness fitness
        fitness.tp_benefit = tp_benefit
        fitness.tn_benefit = tn_benefit
        fitness.fp_cost = fp_cost
        fitness.fn_cost = fn_cost
        fitness.expected_max_profit = NULL
        # Kept alive for the whole fit, which reads them through pointers.
        cdef const double[:, ::1] coefficient_parts
        cdef const double[::1] distribution_parameters
        if closed_form is not None:
            coefficient_parts = closed_form.coefficient_parts
            distribution_parameters = closed_form.distribution_parameters
            fitness.expected_max_profit = <ExpectedMaxProfitOfGroups><void*>expected_max_profit_of_groups
            fitness.coefficient_parts = &coefficient_parts[0, 0]
            fitness.n_powers = coefficient_parts.shape[1]
            fitness.lower_bound = closed_form.lower_bound
            fitness.upper_bound = closed_form.upper_bound
            fitness.distribution = closed_form.distribution
            fitness.distribution_parameters = &distribution_parameters[0]

        cdef EvolutionResult result = evolve_forest_native(
            X,
            y,
            fitness,
            pop_size,
            max_depth,
            max_generations,
            min_samples_split,
            min_samples_leaf,
            crossover_rate,
            grow_rate,
            prune_rate,
            mutate_split_rate,
            mutate_value_rate,
            patience,
            tol,
            alpha,
            random_state,
            n_jobs,
            cache_samples,
        )
        self.tree = result.tree
        self.n_generations = result.n_generations

    def predict_proba(self, cnp.ndarray[cnp.float32_t, ndim=2] X):
        if self.tree is NULL:
            raise ValueError("The model has not been fitted yet.")
        X = np.ascontiguousarray(X)
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[cnp.float32_t, ndim=1] probabilities = np.empty(n_samples, dtype=np.float32)
        cdef float[:] probabilities_view = probabilities
        predict_proba_tree(self.tree, X, probabilities_view, n_samples)
        return probabilities

    def predict(self, cnp.ndarray[cnp.float32_t, ndim=2] X):
        if self.tree is NULL:
            raise ValueError("The model has not been fitted yet.")
        X = np.ascontiguousarray(X)
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[cnp.float32_t, ndim=1] predictions = np.empty(n_samples, dtype=np.float32)
        cdef float[:] predictions_view = predictions
        predict_labels_tree(self.tree, X, predictions_view, n_samples)
        return predictions

    def __dealloc__(self):
        if self.tree is not NULL:
            free_tree(self.tree)

    def __reduce__(self):
        """Enable pickling of EvolutionaryTree."""
        if self.tree is NULL:
            return (self.__class__, (), None)

        tree_data = self._serialize_tree()
        state = (tree_data, self.n_generations)
        return (self.__class__, (), state)

    def __setstate__(self, state):
        """Restore the tree from pickled data."""
        if state is not None:
            tree_data, n_generations = state
            self._deserialize_tree(tree_data)
            self.n_generations = n_generations

    def _serialize_tree(self):
        """Convert the tree structure to a serializable format."""
        if self.tree is NULL:
            return None
        return serialize_tree(self.tree)

    def _deserialize_tree(self, tree_data):
        """Reconstruct the tree from serialized data."""
        if self.tree is not NULL:
            free_tree(self.tree)
        self.tree = deserialize_tree(tree_data)