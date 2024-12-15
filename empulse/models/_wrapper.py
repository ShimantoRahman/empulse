from collections import defaultdict


class WrapperMixin:

    def __getattr__(self, attr):
        """
        If the attribute is not found in class,
        it checks if it is present in `self.estimator_` and if it is, returns that.
        """
        if hasattr(self, 'estimator_') and hasattr(self.estimator_, attr):
            return getattr(self.estimator_, attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`sklearn:sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        valid_params_model = self.estimator.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params and key not in valid_params_model:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )
            if key not in valid_params and key in valid_params_model:
                self.estimator.set_params(**{key: value})
                self.params[key] = value
                continue

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
