"""Class for candidate selectors."""
import abc
from typing import Iterable, Optional

from spacy.kb import Candidate
from spacy.kb_new.kb_interface import KnowledgeBase
from spacy.tokens import Span
from .. import Errors


class CandidateSelector(abc.ABC):
    """Callable object for candidate selection."""

    def __init__(self, kb: Optional[KnowledgeBase]):
        """Creates a candidate selector.
        kb (Optional[KnowledgeBase]): Instance of KnowledgeBase to associate with this selector. Has to be set before
            first execution of __call__().
        """
        self._kb = kb

    @property
    def kb(self) -> Optional[KnowledgeBase]:
        """Get KnowledgeBase instance.
        RETURNS (Optional[KnowledgeBase): Associated instance of KnowledgeBase.
        """
        return self._kb

    @kb.setter
    def kb(self, value: KnowledgeBase) -> None:
        """Set KnowledgeBase instance.
        value (KnowledgeBase): Instance of KnowledgeBase to associate with this selector.
        """
        self._kb = value

    @property
    def _is_initialized(self) -> bool:
        """Checks whether selector has been fully initialized.
        RETURNS (bool): Whether selector has been fully initialized.
        """
        return self._kb is not None

    def __call__(
        self, spans: Iterable[Span], **kwargs
    ) -> Iterable[Iterable[Candidate]]:
        """Identifies entity candidates in text spans.
        dataset_id (str): ID of dataset for which to select candidates.
        span (Span): Span to match potential entity candidates with.
        RETURNS (Iterable[Iterable[Candidate]]): Candidates for specified entities.
        """
        if not self._is_initialized:
            raise ValueError(Errors.E1044)

        return self._select_candidates(spans, **kwargs)

    @abc.abstractmethod
    def _select_candidates(
        self, span: Iterable[Span], **kwargs
    ) -> Iterable[Iterable[Candidate]]:
        """Fetches candidates for entity in span.text.
        span (Span): candidate span.
        RETURNS (Iterable[Iterable[Candidate]]): Candidates for specified entities.
        """
