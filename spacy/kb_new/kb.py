import abc
from typing import List, Iterable, Iterator, Tuple
from spacy import Vocab
from ..tokens import Span
from ..util import SimpleFrozenList


class Candidate(abc.ABC):
    """A `Candidate` object refers to a textual mention (`alias`) that may or may not be resolved
    to a specific `entity` from a Knowledge Base. This will be used as input for the entity linking
    algorithm which will disambiguate the various candidates to the correct one.
    Each candidate (alias, entity) pair is assigned to a certain prior probability.

    DOCS: https://spacy.io/api/kb/#candidate_init
    """

    def __init__(
        self,
        kb: "KnowledgeBase",
        entity: str,
        entity_freq: int,
        entity_vector: List[float],
        alias: str,
        prior_prob: float,
    ):
        """Constructs new instance of Candidate.
        entity (str): Entity ID/name.
        entity_freq (int): Entity frequency.
        entity_vector (List[float]): Entity embedding vector.
        alias (str): Alias ID/name.
        prior_prob (float): Prior probability of
        """
        self.kb = kb
        self._entity = entity
        self._entity_freq = entity_freq
        self._entity_vector = entity_vector
        self._alias = alias
        self._prior_prob = prior_prob

    @property
    def entity_(self) -> str:
        """RETURNS (str): ID/name of this entity in the KB."""
        return self._entity

    @property
    def alias_(self):
        """RETURNS (str): ID of the original alias."""
        return self._alias

    @property
    def entity_freq(self) -> int:
        """RETURNS (int): Entity frequency."""
        return self._entity_freq

    @property
    def entity_vector(self) -> List[float]:
        """RETURNS (List[float]): Entity embedding vector."""
        return self.entity_vector

    @property
    def prior_prob(self) -> float:
        """RETURNS (float): Prior alias-entity probability."""
        return self.prior_prob


class KnowledgeBase(abc.ABC):
    """A KnowledgeBase implements everything necessary in order to leverage a specific data structure for (1) adding or
    modifying existing entities and (2) looking up candidate entities for given mentions.
    A "data structure" can be pretty much anything - custom or third-party, in-memory or on-disk or only available via
    network, relational DB or graph DB or text search engine - as long as the necessary interface is fully implemented.
    It is also possible to implement a data structure directly in a KnowledgeBase class.

    DOCS: https://spacy.io/api/kb.
    """

    def __init__(
        self,
        vocab: Vocab,
        entity_vector_length: int,
    ):
        """Create a KnowledgeBase.
        vocab (Vocab): Model vocabulary.
        entity_vector_length (int): Length of entity embedding vectors.
        """
        self.vocab = vocab
        self._entity_vector_length = entity_vector_length

    @property
    def entity_vector_length(self) -> int:
        """Return entity vector length.
        RETURNS (int): Entity vector length.
        """
        return self._entity_vector_length

    @abc.abstractmethod
    def get_candidates(self, spans: Iterable[Span]) -> Iterable[Iterator[Candidate]]:
        """
        Return candidate entities for specified texts. Each candidate defines the entity, the original alias,
        and the prior probability of that alias resolving to that entity.
        If the no candidate is found for a given text, an empty list is returned.
        spans (Iterable[Span]): Spans for which to get candidates.
        RETURNS (Iterable[Iterator[Candidate]]): Identified candidates.
        """

    @abc.abstractmethod
    def get_vectors(self, entities: Iterable[str]) -> Iterable[List[float]]:
        """
        Return vectors for entities.
        entity (str): Entity name/ID.
        RETURNS (List[float]): Vectors for specified entities.
        """

    @abc.abstractmethod
    def to_bytes(self, **kwargs) -> bytes:
        """Serialize the current state to a binary string.
        RETURNS (bytes): Current state as binary string.
        """

    @abc.abstractmethod
    def from_bytes(self, bytes_data: bytes, *, exclude: Tuple[str] = tuple()):
        """Load state from a binary string.
        bytes_data (bytes): KB state.
        exclude (Tuple[str]): Properties to exclude when restoring KB.
        """

    @abc.abstractmethod
    def to_disk(self, path: str, exclude: Iterable[str] = SimpleFrozenList()) -> None:
        """
        Write KnowledgeBase content to disk.
        path (str): Target file path.
        exclude (Iterable[str]): List of components to exclude.
        """

    @abc.abstractmethod
    def from_disk(self, path: str, exclude: Iterable[str] = SimpleFrozenList()) -> None:
        """
        Load KnowledgeBase content from disk.
        path (str): Target file path.
        exclude (Iterable[str]): List of components to exclude.
        """
