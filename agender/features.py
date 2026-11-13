"""
Encoding of the released attributes for Steps 2B and 3.

Same rules as the baseline models (log(1+x) and standardisation for skewed numbers, one-hot for
categorical columns with few values, frequency encoding for the others), but every encoded column
remembers the attribute it came from. A release is a set of attributes, so Step 3 can drop whole
attributes and keep the rest unchanged.

Everything is fitted on the training rows only.
"""
import numpy as np
from pandas.api.types import is_numeric_dtype

ONE_HOT_LIMIT = 10


class AttributeEncoder:
    def fit(self, table, attributes):
        self.attributes = list(attributes)
        self.kind, self.settings = {}, {}
        for name in self.attributes:
            values = table[name]
            if is_numeric_dtype(values):
                values = values.astype(float)
                use_log = bool(values.min() >= 0 and values.skew() > 1)
                x = np.log1p(values) if use_log else values
                self.kind[name] = "number"
                # percentiles 1..99 of the training values, for binning in codes()
                self.settings[name] = (use_log, float(x.mean()), float(x.std()) or 1.0,
                                       np.quantile(x, np.arange(1, 100) / 100))
            else:
                values = values.fillna("unknown").astype(str)
                if values.nunique() <= ONE_HOT_LIMIT:
                    self.kind[name] = "one_hot"
                    self.settings[name] = sorted(values.unique())
                else:
                    self.kind[name] = "frequency"
                    self.settings[name] = values.value_counts(normalize=True).to_dict()
        return self

    def _number(self, table, name):
        use_log, mean, std, _ = self.settings[name]
        x = table[name].astype(float).to_numpy()
        return np.log1p(x) if use_log else x

    def transform(self, table, attributes=None):
        """Encoded matrix and, for each column, the attribute it belongs to."""
        columns, groups = [], []
        for name in attributes if attributes is not None else self.attributes:
            if self.kind[name] == "number":
                _, mean, std, _ = self.settings[name]
                columns.append((self._number(table, name) - mean) / std)
                groups.append(name)
            elif self.kind[name] == "one_hot":
                values = table[name].fillna("unknown").astype(str)
                for category in self.settings[name]:
                    columns.append((values == category).to_numpy(dtype=float))
                    groups.append(name)
            else:
                values = table[name].fillna("unknown").astype(str)
                columns.append(values.map(self.settings[name]).fillna(0.0).to_numpy(dtype=float))
                groups.append(name)
        return np.column_stack(columns), groups

    def codes(self, table, name, bins=None):
        """Small integer codes of one attribute, for mutual-information estimates.

        Numbers: training-set quantile bins (10 by default, fewer when `bins` is given).
        Categories: rank of the category in the training data; categories unseen in training share one code."""
        if self.kind[name] == "number":
            bins = bins or 10
            percentiles = self.settings[name][3]
            edges = np.unique(percentiles[[round(100 * j / bins) - 1 for j in range(1, bins)]])
            return np.searchsorted(edges, self._number(table, name), side="right")
        values = table[name].fillna("unknown").astype(str)
        order = (self.settings[name] if self.kind[name] == "one_hot"
                 else sorted(self.settings[name], key=self.settings[name].get, reverse=True))
        rank = {category: i for i, category in enumerate(order)}
        out = values.map(rank).fillna(len(order)).astype(int).to_numpy()
        if bins is not None:
            out = np.minimum(out, bins - 1)           # rare categories merged into one "other" code
        return out
