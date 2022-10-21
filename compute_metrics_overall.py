import sys
import json
import argparse
from datasets import load_metric
from tqdm import tqdm
import pprint

parser = argparse.ArgumentParser(
    description="Training abstractive slides generation model."
)
parser.add_argument(
    "--predictions",
    help="Path to the json file containing predictions.",
    required=True,
    type=str,
)

parser.add_argument(
    "--references",
    help="Path to the json file containing references.",
    required=True,
    type=str,
)

args = parser.parse_args()

rouge = load_metric("rouge")
bert_score = load_metric("bertscore")


def getScores(model_out: [], reference: []):
    r = rouge.get_scores(model_out, reference, avg=True)

    rouge_output1 = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge1"]
    )["rouge1"].mid

    rouge_output2 = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    rouge_output1 = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge1"]
    )["rougeL"].mid

    rouge_output2 = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rougeLsum"].mid

    results = bertscore.compute(predictions=model_out, references=reference)

    return r


def finalScore(score: {}, count):
    rouges = ["rouge-1", "rouge-2", "rouge-l"]
    attrs = ["f", "p", "r"]
    d = {}

    for r in rouges:
        d[r] = {}
        for att in attrs:
            tot = score[r][att]
            avg = tot / count
            d[r][att] = avg

    return d

def get_references_dict(file_path):
    with open(file_path) as f:
        test_d = json.load(f)

    references = {}
    for k, v in test_d.items():
        
        references[k] = {}
        references[k]["id"] = k
        
        for e in v["slides"]:
            
            if e["slide_imrad"] == "I":
                if "introduction_o" not in references[k].keys():
                    references[k]["introduction_o"] = "\n".join(e["text"])
                else:
                    references[k]["introduction_o"] += "\n".join(e["text"])
            elif e["slide_imrad"] == "M":
                if "method_o" not in references[k].keys():
                    references[k]["method_o"] = "\n".join(e["text"])
                else:
                    references[k]["method_o"] += "\n".join(e["text"])
            elif e["slide_imrad"] == "R":
                if "result_o" not in references[k].keys():
                    references[k]["result_o"] = "\n".join(e["text"])
                else:
                    references[k]["result_o"] += "\n".join(e["text"])
            elif e["slide_imrad"] == "D":
                if "conclusion_o" not in references[k].keys():
                    references[k]["conclusion_o"] = "\n".join(e["text"])
                else:
                    references[k]["conclusion_o"] += "\n".join(e["text"])

        for im_sec in ["introduction_o", "method_o", "result_o", "conclusion_o"]:
            if im_sec not in references[k].keys():
                references[k][im_sec] = ""

    return references

def get_predictions_dict(file_path):
    with open(file_path) as f:
        pred_list = json.load(f)["predictions"]

    predictions = {}
    for e in pred_list:
        predictions[e["id"]] = {}
        for k, v in e.items():
            predictions[e["id"]][k] = v

    return predictions

if __name__ == "__main__":

    imrad = ["introduction", "method", "result", "conclusion"]

    model = args.predictions  ##model output

    references = get_references_dict(args.references)
    predictions = get_predictions_dict(args.predictions)

    complete_str_res = ""

    d = {}
    count = 0
    score = {}
    score["rouge-1"] = {}
    score["rouge-1"]["f"] = 0
    score["rouge-1"]["p"] = 0
    score["rouge-1"]["r"] = 0
    score["rouge-2"] = {}
    score["rouge-2"]["f"] = 0
    score["rouge-2"]["p"] = 0
    score["rouge-2"]["r"] = 0
    score["rouge-l"] = {}
    score["rouge-l"]["f"] = 0
    score["rouge-l"]["p"] = 0
    score["rouge-l"]["r"] = 0

    list_pred_text = []
    list_ref_text = []

    for k in tqdm(references.keys()):
        predicted_text = ""
        for imrad_section in imrad:
            predicted_text += predictions[k][imrad_section + "_pred"] + " "

        predicted_text = predicted_text.strip()

        reference_text = ""
        for imrad_section in imrad:
            reference_text += references[k][imrad_section + "_o"] + " "

        reference_text = reference_text.strip()

        list_pred_text.append(predicted_text)
        list_ref_text.append(reference_text)

    r_out = rouge.compute(
        predictions=list_pred_text, references=list_ref_text, use_stemmer=False
    )

    b_out = bert_score.compute(
        predictions=list_pred_text, references=list_ref_text,  model_type="distilbert-base-uncased"
    )

    for k, v in r_out.items():
        score = v.mid
        print(f"{k}\tPrecision:\t{score.precision * 100 :.2f}")
        print(f"{k}\tRecall   :\t{score.recall * 100 :.2f}")
        print(f"{k}\tF-measure:\t{score.fmeasure * 100 :.2f}")
        print(k)
        print(f"{score.precision * 100 :.2f}\t{score.recall * 100 :.2f}\t{score.fmeasure * 100 :.2f}")
        complete_str_res += f"{score.precision * 100 :.2f}\t{score.recall * 100 :.2f}\t{score.fmeasure * 100 :.2f}\t"
        
        print("\n\n")

    print ("BERT-score")
    for k, v in b_out.items():
        if k in ["precision", "recall", "f1"]:
            print(f"{k}\t{sum(v)/len(v) * 100 :.2f}")
    
    print(f'{sum(b_out["precision"])/len(b_out["precision"]) * 100 :.2f}\t{sum(b_out["recall"])/len(b_out["recall"]) * 100 :.2f}\t{sum(b_out["f1"])/len(b_out["f1"]) * 100 :.2f}')
    complete_str_res += f'{sum(b_out["precision"])/len(b_out["precision"]) * 100 :.2f}\t{sum(b_out["recall"])/len(b_out["recall"]) * 100 :.2f}\t{sum(b_out["f1"])/len(b_out["f1"]) * 100 :.2f}'

    print("\n\n")

    print(complete_str_res)


"""
python compute_metrics.py --predictions predictions/sciduet/0-512_sciduet_allenai__led-base-16384.json \
    --references Abstractive-Summarization/SciDuet/train-test-split/clean_test.json
"""
