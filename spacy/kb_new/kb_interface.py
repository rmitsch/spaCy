"""Interface for KnowledgeBase for entity or concept linking.
"""
import abc
from typing import List, Any, Iterable, Iterator, Collection
from spacy.kb import Candidate
from spacy.util import SimpleFrozenList


class KnowledgeBase(abc.ABC):
    """A KnowledgeBase implements everything necessary in order to leverage a specific data structure for (1) adding or
    modifying existing entities and (2) looking up candidate entities for given mentions.
    A "data structure" can be pretty much anything - custom or third-party software, in-memory or on-disk
    or only available via network, relational DB or graph DB or text search engine - as long as the necessary interface
    is fully implemented. It is also possible to implement a data structure directly in a KnowledgeBase class.

    DOCS: https://spacy.io/api/kb.
    """

    @abc.abstractmethod
    @property
    def entity_vector_length(self) -> int:
        """Returns entity vector length.
        RETURNS (int): Entity vector length.
        """

    @abc.abstractmethod
    def add_vector(self, entity_vector: List[float]) -> Any:
        """Add an entity vector to the vectors table.
        entity_vector (List[float]): Entity vector to add.
        RETURNS (int): Vector index.
        """

    @abc.abstractmethod
    def add_entity(
        self, entity_hash: int, freq: float, vector_index: Any, feats_row: int
    ) -> Any:
        """Add an entry to the vector of entries.
        entity_hash (int): Hash for entity to add.
        freq (float): Entity frequency.
        vector_index (Any): Index of entity vector.
        RETURNS (Any): Entity index.
        """

    @abc.abstractmethod
    def add_aliases(
        self, alias_hash: int, entry_indices: List[Any], probs: List[float]
    ) -> Any:
        """Connect a mention to a list of potential entities with their prior probabilities .
        After calling this method, make sure to update also the _alias_index using the return value.
        alias_hash (int): Hash for alias to add.
        entry_indices (List[Any]): List of indices of entities to add.
        RETURNS (Any): Alias index.
        """

    @abc.abstractmethod
    def add_alias(
        self, alias: str, entities: List[str], probabilities: List[float]
    ) -> Any:
        """
        For a given alias, add its potential entities and prior probabilies to the KB.
        alias (str): Alias to add.
        entities (List[str]): List of entities this alias refers to.
        probabilities (List[float]): Prior probabilities of the alias referring to the i-th entity.
        RETURN (Any): Hash for alias.
        """

    @abc.abstractmethod
    def set_entities(
        self,
        entity_list: List[str],
        freq_list: List[float],
        vector_list: List[List[float]],
    ) -> None:
        """Sets all entities.
        entity_list (List[str]): List of all entity names.
        freq_list (List[float]): List of entities' frequencies.
        vector_list (List[List[float]]): List of entities' vectors.
        """

    @abc.abstractmethod
    def get_size_entities(self) -> int:
        """Gets number of entities.
        RETURNS (int): Number of entities in KB.
        """

    @abc.abstractmethod
    def get_entity_strings(self) -> Iterable[str]:
        """Gets all entity strings.
        RETURNS (int): Iterable over all entity strings.
        """

    @abc.abstractmethod
    def get_size_aliases(self) -> int:
        """Gets alias count.
        RETURNS (int): Number of aliases.
        """

    @abc.abstractmethod
    def get_alias_strings(self) -> Iterable[str]:
        """Gets all alias strings.
        RETURNS (int): Iterable over all alias strings.
        """

    @abc.abstractmethod
    def contains_entity(self, entity: str) -> bool:
        """Determines whether KB contains entity.
        RETURNS (bool): Whether KB contains entity.
        """

    @abc.abstractmethod
    def contains_alias(self, alias: str) -> bool:
        """Determines whether KB contains alias.
        RETURNS (bool): Whether KB contains alias.
        """

    @abc.abstractmethod
    def append_alias(
        self, alias: str, entity: str, prior_prob: float, ignore_warnings: bool = False
    ) -> None:
        """
        For an alias already existing in the KB, extend its potential entities with one more.
        Throw a warning if either the alias or the entity is unknown,
        or when the combination is already previously recorded.
        Throw an error if this entity+prior prob would exceed the sum of 1.
        For efficiency, it's best to use the method `add_alias` as much as possible instead of this one.
        alias (str): Alias to append.
        entity (str): Entity name/ID.
        prior_prob (float): Alias-entity prior probability.
        ignore_warnings (bool): Whether to ignore warnings about the alias already being present in the KB.
        """

    @abc.abstractmethod
    def get_alias_candidates(self, alias: str) -> Iterator[Candidate]:
        """
        Return candidate entities for an alias. Each candidate defines the entity, the original alias,
        and the prior probability of that alias resolving to that entity.
        If the alias is not known in the KB, and empty list is returned.
        alias (str): Alias for which to get candidates.
        RETURNS (Iterator[Candidate]): Identified candidates for this alias.
        """

    def get_aliases_candidates(
        self, aliases: Iterable[str]
    ) -> Iterable[Iterator[Candidate]]:
        """
        Return candidate entities for specified aliases. Each candidate defines the entity, the original alias,
        and the prior probability of that alias resolving to that entity.
        If the alias is not known in the KB, and empty list is returned.
        aliases (Iterable[str]): Aliases for which to get candidates.
        RETURNS (Iterable[Iterator[Candidate]]): Identified candidates for these aliases.
        """

    @abc.abstractmethod
    def get_vector(self, entity: str) -> List[float]:
        """
        Return vector for entity.
        entity (str): Entity name/ID.
        RETURNS (List[float]): Vector for specified entity.
        """

    @abc.abstractmethod
    def get_vectors(self, entities: Iterable[str]) -> Iterable[List[float]]:
        """
        Return vectors for entities.
        entities (str): Entity names/IDs.
        RETURNS (Iterable[List[float]]): Vector for specified entities.
        """

    @abc.abstractmethod
    def get_prior_prob(self, entity: str, alias: str) -> float:
        """
        Return the prior probability of a given alias being linked to a given entity,
        or return 0.0 when this combination is not known in the knowledge base
        entity (str): Entity name/ID.
        alias (str): Alias.
        RETURNS (float): Prior probability of alias for entity.
        """

    @abc.abstractmethod
    def get_prior_probs(
        self, entities: Collection[str], aliases: Collection[str]
    ) -> Iterable[float]:
        """
        Return the prior probability of a given alias being linked to a given entity,
        or return 0.0 when this combination is not known in the knowledge base
        entities (Collection[str]): Entity names/IDs.
        aliases (Collection[str]): Aliases.
        RETURNS (float): Prior probability of alias for entity.
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
