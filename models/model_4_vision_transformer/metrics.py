import wandb
import os
from typing import List, Dict
import torch
import evaluate
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from transformers import (
    GPT2TokenizerFast,
    VisionEncoderDecoderModel,
)
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from transformers.generation import GreedySearchEncoderDecoderOutput
from tqdm import tqdm
from torchvision.transforms import transforms
from PIL import Image
from typing import List, Dict


def calculate_metrics(
    model: VisionEncoderDecoderModel,
    image_dir: str,
    image_ids: List[str],
    image2captions: Dict[str, List[str]],
    transform: transforms.Compose,
    tokenizer: GPT2TokenizerFast,
    device: torch.device,
    max_length: int = 50,
    verbose: bool = False,
    enable_wandb: bool = False,
) -> Dict[str, float]:
    meteor_metric = evaluate.load("meteor")
    cider_scorer = Cider()

    model.eval()
    with torch.no_grad():
        references: List[List[str]] = []
        hypotheses: List[str] = []
        sample_preds: List[str] = []
        sample_refs: List[List[str]] = []
        processed_image_ids: List[str] = []
        gts: Dict[str, List[Dict[str, str]]] = {}
        res: Dict[str, List[Dict[str, str]]] = {}

        if enable_wandb:
            sample_table = wandb.Table(columns=["Image", "Prediction", "References"])
        else:
            sample_table = None

        for i, image_id in enumerate(tqdm(image_ids, desc="Eval Metrics..")):
            img_path = os.path.join(image_dir, image_id)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            # print(f"running inference on image: {img_path}")
            # Forward pass
            output: GreedySearchEncoderDecoderOutput = model.generate(
                pixel_values=image,
                max_length=max_length,
                num_beams=1,
                do_sample=False,
                return_dict_in_generate=True,
            )

            # print(f"Output: {output}")
            pred = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
            # print(f"Prediction: {pred}")
            hypotheses.append(pred)
            refs = image2captions.get(
                image_id, []
            )  # list of (list of strings (captions))
            processed_image_ids.append(image_id)

            if verbose and len(sample_preds) < 5:
                sample_preds.append(pred)
                sample_refs.append(refs)

                if enable_wandb:
                    # Add the image, prediction, and references to the table
                    sample_table.add_data(wandb.Image(image[0]), pred, "\n".join(refs))

            refs_flat: List[str] = [
                " ".join(
                    [
                        word.lower()
                        for word in ref
                        # image2captions was processed with our own special tokens
                        # and ntlk tokenizer
                        # and not using gpt2 tokenizer
                        # we just want to ensure that the words are not bpe-ed for fair comparison
                    ]
                )
                for ref in refs
            ]
            references.append(refs_flat)
            gts[image_id] = [{"caption": ref} for ref in refs_flat]
            res[image_id] = [{"caption": pred}]

        # print(
        #     f"References: {references}, Hypotheses: {hypotheses}, Image IDs: {processed_image_ids}"
        # )
        # print(f"gts: {gts}, res: {res}")
        if enable_wandb:
            wandb.log({"Sample Predictions": sample_table})

        # tokenize
        ptb_tokenizer = PTBTokenizer()
        gts = ptb_tokenizer.tokenize(gts)
        res = ptb_tokenizer.tokenize(res)

        # Calculate scores
        cider_score, _ = cider_scorer.compute_score(gts, res)
        # print(f"INPUT to bleu: {references}, {hypotheses}")
        # print(
        #     f"modified input to bleu: REF {[[ref.split() for ref in refs_flat] for refs_flat in references]}, HYP {[hyp.split() for hyp in hypotheses]}"
        # )
        bleu_score = corpus_bleu(
            [[ref.split() for ref in refs_flat] for refs_flat in references],
            [hyp.split() for hyp in hypotheses],
            smoothing_function=SmoothingFunction().method4,
        )
        meteor_score = meteor_metric.compute(
            predictions=hypotheses, references=references
        )["meteor"]

        if verbose:
            print("Sample Predictions:")
            for pred, refs in zip(sample_preds, sample_refs):
                print(f"Prediction: {pred}")
                print(f"References: {refs}")
                print("-" * 80)

        return {
            "bleu": bleu_score,
            "meteor": meteor_score,
            "cider": cider_score,
        }
