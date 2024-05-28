from typing import Any, Dict, List, Optional
from evaluate import load
from summac.model_summac import SummaCZS, SummaCConv

from pruning_study.datamodels import HallucinationResult, SummaryResult



class ExperimentEvaluator:
    """a class for evaluating experiments"""
    def __init__(
        self,
        device: Optional[str] = 'cuda'
    ):

        # summarization functions
        self.rouge =  load('rouge')
        self.bertscore = load('bertscore')


        # hallucination functions
        self.harim_plus = load("NCSOFT/harim_plus")
        self.summac_zs = SummaCZS(
            granularity="sentence",
            model_name="vitc",
            device=device,
            use_con=False
        )
        self.summac_conv = SummaCConv(
            models=["vitc"],
            bins='percentile',
            granularity="sentence",
            nli_labels="e",
            device=device,
            start_file="default",
            agg="mean"
        )


    def evaluate_summary(
        self,
        prediction: str | List[str],
        reference: str | List[str]
    ) -> SummaryResult:
        """evaluates a summary"""

        if isinstance(prediction, str):
            prediction = [prediction]
        if isinstance(reference, str):
            reference = [reference]

        rouge_results: Dict[str, List[float]] = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
        }
        # rouge is good only for a single thing
        # so have to do this
        for indx, single_prediction in enumerate(prediction):

            rouge_single_result: Dict[str, Any] = self.rouge.compute(
                predictions=[single_prediction],references=[reference[indx]]
            )

            rouge_results['rouge1'].append(rouge_single_result['rouge1'])
            rouge_results['rouge2'].append(rouge_single_result['rouge2'])
            rouge_results['rougeL'].append(rouge_single_result['rougeL'])

        # for bertscore its fine
        bertscore_result: Dict[str, List[float] | str] =  self.bertscore.compute(
            predictions=prediction,
            references=reference,
            lang="en"
        )

        return SummaryResult(
            bertscore=bertscore_result,
            rouge=rouge_results
        )

    def evaluate_hallucunations(
        self,
        prediction: str | List[str],
        reference: str | List[str]
    ) -> HallucinationResult:
        """evaluates for hallucinations"""

        if isinstance(prediction, str):
            prediction = [prediction]
        if isinstance(reference, str):
            reference = [reference]

        harim_result: List[float] = self.harim_plus.compute(
            predictions=prediction,
            references=reference
        )

        summac_zs_result: Dict[str, Any] = self.summac_zs.score(
            sources=reference,
            generateds=prediction
        )

        summac_conv_result: Dict[str, Any] = self.summac_conv.score(
            originals=reference,
            generateds=prediction
        )

        return HallucinationResult(
            summac_conv=summac_conv_result['scores'],
            summac_zs=summac_zs_result['scores'],
            harim_plus=harim_result
        )
