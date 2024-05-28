from typing import Optional
from pruning_study.prompts.formatters import TEMPLATE_FORMATTER
from pruning_study.prompts.instructions import TASK_INSTRUCTION
from pruning_study.prompts.roles import TASK_ROLE
import warnings

__all__ = ['generate_prompt']


def generate_prompt(
    task: str, model: str, prompt_id: str, document: str, question: Optional[str] = None, warningon: bool = False
) -> str:
    """formats messages in appropriate prompt format for specific models models"""

    assert task in ['question_answering', 'summarization']
    assert model in ['llama', 'mistral', 'falcon', 'opt']
    assert prompt_id in ['A', 'B', 'C']

    if task == 'summarization':
        if (question is not None)and(warningon):
            warnings.warn("`question` field is not empty. Did you want to use the `question answering` task?")
        instruction = TASK_INSTRUCTION[task](
            prompt_id=prompt_id,
            document = document
        )
    else:
        assert question is not None
        instruction = TASK_INSTRUCTION[task](
            prompt_id=prompt_id,
            document = document,
            question = question
        )

    return TEMPLATE_FORMATTER[model](
        messages=[
            {
                "role": "system",
                "content": TASK_ROLE[task],
            },
            {"role": "user", "content": instruction},
        ],
    )
