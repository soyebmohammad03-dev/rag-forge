"""The grounded prompt contract: versioned, rendered deterministically, recorded verbatim.

The wording was chosen by trying variants against the default 0.5B model on in-corpus and
out-of-corpus questions (docs/rag.md). Longer rule lists and few-shot examples made that model
copy the examples or abstain on answerable questions. This template kept its answers inside the
evidence and made it abstain when the evidence lacked the answer. It still rarely writes
citations, and the grounding report measures that rather than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_forge.domain.models import ChatMessage, ChatRole

INSUFFICIENT = "The evidence is insufficient."


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    system: str
    user: str  # str.format fields: {evidence}, {question}

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    def render(self, question: str, evidence_text: str) -> list[ChatMessage]:
        return [
            ChatMessage(role=ChatRole.SYSTEM, content=self.system),
            ChatMessage(
                role=ChatRole.USER,
                content=self.user.format(evidence=evidence_text, question=question.strip()),
            ),
        ]


GROUNDED_QA = PromptTemplate(
    name="grounded-qa",
    version=1,
    system=(
        "You answer questions using only the evidence passages you are given. "
        "You do not use outside knowledge. "
        "Cite the passage id in square brackets, like [E1], after each sentence, and only cite "
        "ids that appear in the passages. "
        "If the passages only partly answer the question, say which part is uncertain. "
        f'If they do not answer it, say "{INSUFFICIENT}"'
    ),
    user=(
        "Read the passages and answer the question. Cite passages like [E1].\n\n"
        "{evidence}\n\n"
        "Question: {question}\n"
        f'If the passages do not contain the answer, say "{INSUFFICIENT}" Answer:'
    ),
)

TEMPLATES = {GROUNDED_QA.id: GROUNDED_QA}
