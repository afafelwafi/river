# Modified Dataset base class to support weights
from __future__ import annotations

import abc
import inspect
import itertools
import os
import pathlib
import re
import shutil
import tarfile
import typing
import zipfile
from urllib import request
from typing import Iterator, Tuple, Dict, Any, Optional, Union

from river import utils

__all__ = ["Dataset", "SyntheticDataset", "FileDataset", "RemoteDataset"]

REG = "Regression"
BINARY_CLF = "Binary classification"
MULTI_CLF = "Multi-class classification"
MO_BINARY_CLF = "Multi-output binary classification"
MO_REG = "Multi-output regression"


def get_data_home():
    """Return the location where remote datasets are to be stored."""

    data_home = os.environ.get("RIVER_DATA", os.path.join("~", "river_data"))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


class Dataset(abc.ABC):
    """Base class for all datasets.

    All datasets inherit from this class, be they stored in a file or generated on the fly.

    Parameters
    ----------
    task
        Type of task the dataset is meant for. Should be one of the following:

            - "Regression"
            - "Binary classification"
            - "Multi-class classification"
            - "Multi-output binary classification"
            - "Multi-output regression"
    n_features
        Number of features in the dataset.
    n_samples
        Number of samples in the dataset.
    n_classes
        Number of classes in the dataset, only applies to classification datasets.
    n_outputs
        Number of outputs the target is made of, only applies to multi-output datasets.
    sparse
        Whether the dataset is sparse or not.

    """

    def __init__(
        self,
        task,
        n_features,
        n_samples=None,
        n_classes=None,
        n_outputs=None,
        sparse=False,
    ):
        self.task = task
        self.n_features = n_features
        self.n_samples = n_samples
        self.n_outputs = n_outputs
        self.n_classes = n_classes
        self.sparse = sparse
        # Default weight for instances if not specified
        self._default_weight = 1.0

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tuple[Dict, Any]]:
        """Yield instances as (x, y) tuples."""
        raise NotImplementedError

    def iter_with_weights(self) -> Iterator[Tuple[Dict, Any, float]]:
        """Yield instances with weights as (x, y, w) tuples.

        By default, this method yields the default weight (1.0) for each instance.
        Subclasses can override this method to provide custom weights.

        Returns
        -------
        Iterator[Tuple[Dict, Any, float]]
            An iterator over (features, target, weight) tuples.
        """
        for x, y in self:
            yield x, y, self._default_weight

    def with_weights(self, weights=None) -> 'WeightedDataset':
        """Create a weighted version of this dataset.

        Parameters
        ----------
        weights : Union[float, list, dict, callable], optional
            The weights to use for each instance. Can be:
            - A single float value to use the same weight for all instances
            - A list of weights (must have the same length as the dataset if n_samples is known)
            - A dictionary mapping indices to weights
            - A callable that takes (x, y) and returns a weight
            If None, the default weight (1.0) is used for all instances.

        Returns
        -------
        WeightedDataset
            A dataset that yields weighted instances.
        """
        return WeightedDataset(self, weights)

    def take(self, k: int):
        """Iterate over the k samples."""
        return itertools.islice(self, k)

    @property
    def desc(self):
        """Return the description from the docstring."""
        desc = re.split(pattern=r"\w+\n\s{4}\-{3,}", string=self.__doc__, maxsplit=0)[0]
        return inspect.cleandoc(desc)

    @property
    def _repr_content(self):
        """The items that are displayed in the __repr__ method.

        This property can be overridden in order to modify the output of the __repr__ method.

        """

        content = {}
        content["Name"] = self.__class__.__name__
        content["Task"] = self.task
        if isinstance(self, SyntheticDataset) and self.n_samples is None:
            content["Samples"] = "∞"
        elif self.n_samples:
            content["Samples"] = f"{self.n_samples:,}"
        if self.n_features:
            content["Features"] = f"{self.n_features:,}"
        if self.n_outputs:
            content["Outputs"] = f"{self.n_outputs:,}"
        if self.n_classes:
            content["Classes"] = f"{self.n_classes:,}"
        content["Sparse"] = str(self.sparse)

        return content

    def __repr__(self):
        l_len = max(map(len, self._repr_content.keys()))
        r_len = max(map(len, self._repr_content.values()))

        out = f"{self.desc}\n\n" + "\n".join(
            k.rjust(l_len) + "  " + v.ljust(r_len) for k, v in self._repr_content.items()
        )

        if "Parameters\n    ----------" in self.__doc__:
            params = re.split(
                r"\w+\n\s{4}\-{3,}",
                re.split("Parameters\n    ----------", self.__doc__)[1],
            )[0].rstrip()
            out += f"\n\nParameters\n----------{params}"

        return out


class WeightedDataset(Dataset):
    """A dataset wrapper that adds weights to instances.

    Parameters
    ----------
    dataset : Dataset
        The base dataset to wrap.
    weights : Union[float, list, dict, callable], optional
        The weights to use for each instance. Can be:
        - A single float value to use the same weight for all instances
        - A list of weights (must have the same length as the dataset if n_samples is known)
        - A dictionary mapping indices to weights (for sparse weight assignments)
        - A callable that takes (x, y) and returns a weight
        If None, the default weight (1.0) is used for all instances.
    """

    def __init__(
        self,
        dataset: Dataset,
        weights: Union[float, list, dict, callable] = None,
    ):
        super().__init__(
            task=dataset.task,
            n_features=dataset.n_features,
            n_samples=dataset.n_samples,
            n_classes=dataset.n_classes,
            n_outputs=dataset.n_outputs,
            sparse=dataset.sparse,
        )
        self.dataset = dataset
        self.weights = weights

    def __iter__(self):
        """Yield instances as (x, y) tuples."""
        for x, y, _ in self.iter_with_weights():
            yield x, y

    def iter_with_weights(self):
        """Yield instances with weights as (x, y, w) tuples.

        Returns
        -------
        Iterator[Tuple[Dict, Any, float]]
            An iterator over (features, target, weight) tuples.
        """
        weights = self.weights
        
        # If weights is a single float value
        if isinstance(weights, (int, float)):
            for x, y in self.dataset:
                yield x, y, float(weights)
            return
        
        # If weights is a list
        if isinstance(weights, list):
            if self.n_samples and len(weights) != self.n_samples:
                raise ValueError(f"Expected {self.n_samples} weights, got {len(weights)}")
            for (x, y), w in zip(self.dataset, weights):
                yield x, y, float(w)
            return
        
        # If weights is a dictionary mapping indices to weights
        if isinstance(weights, dict):
            for i, (x, y) in enumerate(self.dataset):
                yield x, y, float(weights.get(i, 1.0))
            return
        
        # If weights is a callable that takes (x, y) and returns a weight
        if callable(weights):
            for x, y in self.dataset:
                yield x, y, float(weights(x, y))
            return
            
        # Default: use default weight of 1.0
        for x, y in self.dataset:
            yield x, y, 1.0

    @property
    def _repr_content(self):
        content = super()._repr_content
        content["Name"] = f"WeightedDataset({self.dataset.__class__.__name__})"
        weight_type = type(self.weights).__name__ if self.weights is not None else "default"
        content["Weights"] = weight_type
        return content


class SyntheticDataset(Dataset):
    """A synthetic dataset.

    Parameters
    ----------
    task
        Type of task the dataset is meant for. Should be one of:
        - "Regression"
        - "Binary classification"
        - "Multi-class classification"
        - "Multi-output binary classification"
        - "Multi-output regression"
    n_features
        Number of features in the dataset.
    n_samples
        Number of samples in the dataset.
    n_classes
        Number of classes in the dataset, only applies to classification datasets.
    n_outputs
        Number of outputs the target is made of, only applies to multi-output datasets.
    sparse
        Whether the dataset is sparse or not.

    """

    def __repr__(self):
        l_len_prop = max(map(len, self._repr_content.keys()))
        r_len_prop = max(map(len, self._repr_content.values()))
        params = self._get_params()
        l_len_config = max(map(len, params.keys()))
        r_len_config = max(map(len, map(str, params.values())))

        out = (
            "Synthetic data generator\n\n"
            + "\n".join(
                k.rjust(l_len_prop) + "  " + v.ljust(r_len_prop)
                for k, v in self._repr_content.items()
            )
            + "\n\nConfiguration\n-------------\n"
            + "\n".join(
                k.rjust(l_len_config) + "  " + str(v).ljust(r_len_config) for k, v in params.items()
            )
        )

        return out

    def _get_params(self) -> dict[str, typing.Any]:
        """Return the parameters that were used during initialization."""
        return {
            name: getattr(self, name)
            for name, param in inspect.signature(self.__init__).parameters.items()  # type: ignore
            if param.kind != param.VAR_KEYWORD
        }


class FileDataset(Dataset):
    """Base class for datasets that are stored in a local file.

    Small datasets that are part of the river package inherit from this class.

    Parameters
    ----------
    filename
        The file's name.
    directory
        The directory where the file is contained. Defaults to the location of the `datasets`
        module.
    desc
        Extra dataset parameters to pass as keyword arguments.

    """

    def __init__(self, filename, directory=None, **desc):
        super().__init__(**desc)
        self.filename = filename
        self.directory = directory

    @property
    def path(self):
        if self.directory:
            return pathlib.Path(self.directory).joinpath(self.filename)
        return pathlib.Path(__file__).parent.joinpath(self.filename)

    @property
    def _repr_content(self):
        content = super()._repr_content
        content["Path"] = str(self.path)
        return content


class RemoteDataset(FileDataset):
    """Base class for datasets that are stored in a remote file.

    Medium and large datasets that are not part of the river package inherit from this class.

    The filename doesn't have to be provided if unpack is False. Indeed in the latter case the
    filename will be inferred from the URL.

    Parameters
    ----------
    url
        The URL the dataset is located at.
    size
        The expected download size.
    unpack
        Whether to unpack the download or not.
    filename
        An optional name to given to the file if the file is unpacked.
    desc
        Extra dataset parameters to pass as keyword arguments.

    """

    def __init__(self, url, size, unpack=True, filename=None, **desc):
        if filename is None:
            filename = os.path.basename(url)

        super().__init__(filename=filename, **desc)
        self.url = url
        self.size = size
        self.unpack = unpack

    @property
    def path(self):
        return pathlib.Path(get_data_home(), self.__class__.__name__, self.filename)

    def download(self, force=False, verbose=True):
        if not force and self.is_downloaded:
            return

        # Determine where to download the archive
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        archive_path = directory.joinpath(os.path.basename(self.url))

        with request.urlopen(self.url) as r:
            # Notify the user
            if verbose:
                meta = r.info()
                try:
                    n_bytes = int(meta["Content-Length"])
                    msg = f"Downloading {self.url} ({utils.pretty.humanize_bytes(n_bytes)})"
                except (KeyError, TypeError):
                    msg = f"Downloading {self.url}"
                print(msg)

            # Now dump the contents of the requests
            with open(archive_path, "wb") as f:
                shutil.copyfileobj(r, f)

        if not self.unpack:
            return

        if verbose:
            print(f"Uncompressing into {directory}")

        if archive_path.suffix.endswith("zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(directory)

        elif archive_path.suffix.endswith(("gz", "tar")):
            mode = "r:" if archive_path.suffix.endswith("tar") else "r:gz"
            tar = tarfile.open(archive_path, mode)
            tar.extractall(directory)
            tar.close()

        else:
            raise RuntimeError(f"Unhandled extension type: {archive_path.suffix}")

        # Delete the archive file now that it has been uncompressed
        archive_path.unlink()

    @abc.abstractmethod
    def _iter(self):
        pass

    @property
    def is_downloaded(self):
        """Indicate whether or the data has been correctly downloaded."""
        if self.path.exists():
            if self.path.is_file():
                return self.path.stat().st_size == self.size
            return sum(f.stat().st_size for f in self.path.glob("**/*") if f.is_file())

        return False

    def __iter__(self):
        if not self.is_downloaded:
            self.download(verbose=True)
        if not self.is_downloaded:
            raise RuntimeError("Something went wrong during the download")
        yield from self._iter()

    @property
    def _repr_content(self):
        content = super()._repr_content
        content["URL"] = self.url
        content["Size"] = utils.pretty.humanize_bytes(self.size)
        content["Downloaded"] = str(self.is_downloaded)
        return content