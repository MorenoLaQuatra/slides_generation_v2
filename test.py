import argparse
from utils.utils import *

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import json
from tqdm import tqdm

BT_TOKEN = "<bt>"
ET_TOKEN = "<et>"
BS_TOKEN = "<bs>"
ES_TOKEN = "<es>"
CONTEXT_TOKEN = "<context>"

parser = argparse.ArgumentParser(
    description="Training abstractive slides generation model."
)
parser.add_argument(
    "--trained_model_suffix",
    help="Suffix of the trained model (e.g., _appred_allenai__led-base-16384 ).",
    required=True,
    type=str,
)

parser.add_argument(
    "--trained_models_root",
    help="Path to the folder containing trained models (e.g., models/0-512/ ).",
    required=True,
    type=str,
)

parser.add_argument(
    "--test_set_path",
    help="Path to the test set in json format.",
    required=True,
    type=str,
)

parser.add_argument(
    "--output_file_path",
    help="Path to the output file containing the predictions.",
    required=True,
    type=str,
)

parser.add_argument(
    "--use_global_attention_mask",
    help="Flag to disable the use of context for model training.",
    action="store_true",
)

parser.add_argument(
        "--max_input_length",
        help="Maximum input length for the model",
        default=4096,
        type=int,
    )

parser.add_argument("--use_cuda", help="Flag to enable cuda.", action="store_true")

parser.add_argument("--use_context", help="Flag to enable context.", action="store_true")

args = parser.parse_args()

print(args)

with open(args.test_set_path) as f:
    test_d = json.load(f)

test_data = []
for k, v in test_d.items():
    d = v
    d["id"] = k
    test_data.append(d)

papers_info = {}

for imrad_section in ["introduction", "method", "result", "conclusion"]:

    print (args.trained_model_suffix, imrad_section)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        f"{args.trained_models_root}/{imrad_section}{args.trained_model_suffix}/best_checkpoint/"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        f"{args.trained_models_root}/{imrad_section}{args.trained_model_suffix}/best_checkpoint/"
    )
    if args.use_cuda:
        model = model.to("cuda")

    for e in tqdm(test_data):


        input_section_text = ""
        input_context_text = ""
        output_text = ""

        for sec in e["sections"]:
            sec_imrad = sec["section_imrad"]
            if sec_imrad == "D": sec_imrad = "C" #adjust for conclusion

            if sec_imrad == "A":
                # Sentences
                for s in sec["text"]:
                    input_context_text += BS_TOKEN + s + ES_TOKEN
            elif sec_imrad == imrad_section[0].upper():
                # Explicit title
                input_section_text += BT_TOKEN + sec["heading"] + ET_TOKEN
                # Sentences
                for s in sec["text"]:
                    if s != "":
                        input_section_text += BS_TOKEN + s + ES_TOKEN
            
        for slide in e["slides"]:
            slide_imrad = slide["slide_imrad"]
            if slide_imrad == "D": slide_imrad = "C" #adjust for conclusion

            if slide_imrad == imrad_section[0].upper():
                output_text += BT_TOKEN + slide["heading"] + ET_TOKEN
                for s in slide["text"]:
                    if s != "":
                        output_text += BS_TOKEN + s + ES_TOKEN
        
        if args.use_context:
            input_text = input_section_text + CONTEXT_TOKEN + input_context_text


        inputs = tokenizer.encode_plus(
            input_text,
            padding="max_length",
            max_length=args.max_input_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )


        if args.use_global_attention_mask:
            global_attention_mask = torch.zeros(len(inputs["input_ids"]))
            global_attention_mask[0] = 1
            
        if args.use_cuda:
            input_ids = inputs["input_ids"].to("cuda")
            if args.use_global_attention_mask:
                global_attention_mask = global_attention_mask.to("cuda")
        else:
            input_ids = inputs["input_ids"]

        if args.use_global_attention_mask:
            summary_ids = model.generate(
                input_ids,
                num_return_sequences=1,
                num_beams=3,
                global_attention_mask=global_attention_mask,
                early_stopping=True,
            )
        else:
            summary_ids = model.generate(
                input_ids,
                num_return_sequences=1,
                num_beams=3,
                early_stopping=True,
            )

        summary_text = [
            tokenizer.decode(
                g, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            for g in summary_ids
        ]

        summary_text = [e.replace(f"{ES_TOKEN}{BS_TOKEN}", "\n") for e in summary_text]
        summary_text = [e.replace(f"{ET_TOKEN}{BS_TOKEN}", "\n") for e in summary_text]
        summary_text = [e.replace(ES_TOKEN, "") for e in summary_text]
        summary_text = [e.replace(BS_TOKEN, "") for e in summary_text]
        summary_text = [e.replace(BT_TOKEN, "") for e in summary_text]
        summary_text = [e.replace("<pad>", "") for e in summary_text]
        summary_text = [e.replace("<s>", "") for e in summary_text]
        summary_text = [e.replace("</s>", "") for e in summary_text]

        summary_text = " ".join(summary_text)

        print (summary_text)

        if e["id"] in papers_info.keys():
            papers_info[e["id"]][imrad_section + "_pred"] = summary_text
        else:
            papers_info[e["id"]] = {}
            papers_info[e["id"]][imrad_section + "_pred"] = summary_text

    print("\n*********************************\n\n")
predictions = []
for k, v in papers_info.items():
    d = {
        "id": k,
        "introduction_pred": v["introduction_pred"],
        "method_pred": v["method_pred"],
        "result_pred": v["result_pred"],
        "conclusion_pred": v["conclusion_pred"],
    }
    predictions.append(d)

with open(args.output_file_path, "w") as f:
    f.write(json.dumps({"predictions": predictions}, indent=4))

"""
python test_appred.py \
    --trained_model_suffix _sciduet_allenai__led-base-16384 \
    --trained_models_root models/0_512/ \
    --test_set_path Abstractive-Summarization/SciDuet/train-test-split/clean_test.json \
    --output_file_path predictions/sciduet/0-512_sciduet_allenai__led-base-16384.json
"""
