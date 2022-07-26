"""Knowledge base for entity or concept linking."""
import abc
from typing import List, Iterable, Iterator, Collection, Optional
from spacy.kb import Candidate
from spacy import Vocab
from .candidate_selector import CandidateSelector
from . import kb_interface
from ..tokens import Span


class KnowledgeBase(kb_interface.KnowledgeBase, abc.ABC):
    """Implementation of the KnowledgeBase interface.
    DOCS: https://spacy.io/api/kb.
    """

    def __init__(
        self,
        vocab: Vocab,
        entity_vector_length: int,
        candidate_selector: Optional[CandidateSelector],
    ):
        """Create a KnowledgeBase.
        vocab (Vocab): Model vocabulary.
        entity_vector_length (int): Length of entity embedding vectors.
        candidate_selector (CandidateSelector): CandidateSelector instance to use for looking up entity candidates. If
            None, [components.entity_linker.get_candidates] has to have an implementation other than
            `KnowledgeBase.get_alias_candidates()` / `KnowledgeBase.get_aliases_candidates()`.
        """
        self.vocab = vocab
        self._entity_vector_length = entity_vector_length
        self._candidate_selector = candidate_selector
        if self._candidate_selector:
            self._candidate_selector.kb = self

    @property
    def entity_vector_length(self) -> int:
        return self._entity_vector_length

    def get_candidates(self, spans: Iterable[Span]) -> Iterable[Iterator[Candidate]]:
        # This will fail if (1) no candidate selector has been supplied and (2) get_candidates() in the config has not
        # been explicitly linked with a different function.
        return self._candidate_selector(spans)

    def get_vectors(self, entities: Iterable[str]) -> Iterable[List[float]]:
        return [self.get_vector(entity) for entity in entities]

    def get_prior_probs(
        self, entities: Collection[str], aliases: Collection[str]
    ) -> Iterable[float]:
        assert len(entities) == len(aliases)
        return [
            self.get_prior_prob(entity, alias)
            for entity, alias in zip(entities, aliases)
        ]
