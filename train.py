from utils.utils import *

import torch
import json
from torch.utils.data import Dataset
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

BT_TOKEN = "<bt>"
ET_TOKEN = "<et>"
BS_TOKEN = "<bs>"
ES_TOKEN = "<es>"
CONTEXT_TOKEN = "<context>"

# TODO: pre-training to reconstruct sections!!

args = parse_arguments()

with open(args.train_set_path) as f:
    train_d = json.load(f)

with open(args.val_set_path) as f:
    val_d = json.load(f)

with open(args.test_set_path) as f:
    test_d = json.load(f)

train_data = []
for k, v in train_d.items():
    d = v
    d["id"] = k
    train_data.append(d)

val_data = []
for k, v in val_d.items():
    d = v
    d["id"] = k
    val_data.append(d)

test_data = []
for k, v in test_d.items():
    d = v
    d["id"] = k
    test_data.append(d)

class InternalDataset(Dataset):
    def __init__(
        self,
        elements,
        imrad_section,
        max_input_length,
        max_output_length,
        tokenizer=None,
        use_global_attention_mask=True,
    ):
        super(InternalDataset, self).__init__()
        self.elements = elements
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.imrad_section = imrad_section
        self.use_global_attention_mask = use_global_attention_mask

    def __getitem__(self, idx):

        """
        print (self.imrad_section + "_i")
        print (self.elements[idx][self.imrad_section + "_i"])
        print ("abstract")
        print (self.elements[idx]["abstract"])
        print ()
        print (self.imrad_section + "_o")
        print (self.elements[idx][self.imrad_section + "_o"])
        print ("----------------------------------\n\n\n\n")
        """


        input_section_text = ""
        input_context_text = ""
        output_text = ""

        for sec in self.elements[idx]["sections"]:
            sec_imrad = sec["section_imrad"]
            if sec_imrad == "D": sec_imrad = "C" #adjust for conclusion

            if sec_imrad == "A":
                # Sentences
                for s in sec["text"]:
                    input_context_text += BS_TOKEN + s + ES_TOKEN
            elif sec_imrad == self.imrad_section[0].upper():
                # Explicit title
                input_section_text += BT_TOKEN + sec["heading"] + ET_TOKEN
                # Sentences
                for s in sec["text"]:
                    if s != "":
                        input_section_text += BS_TOKEN + s + ES_TOKEN
            
        for slide in self.elements[idx]["slides"]:
            slide_imrad = slide["slide_imrad"]
            if slide_imrad == "D": slide_imrad = "C" #adjust for conclusion

            if slide_imrad == self.imrad_section[0].upper():
                output_text += BT_TOKEN + slide["heading"] + ET_TOKEN
                for s in slide["text"]:
                    if s != "":
                        output_text += BS_TOKEN + s + ES_TOKEN

        if args.use_context:
            input_text = input_section_text + CONTEXT_TOKEN + input_context_text

        item = {}
        out = self.tokenizer.encode_plus(
                input_text,
                padding="max_length",
                max_length=self.max_input_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
                return_token_type_ids=True,
            )
        

        item["input_ids"] = out["input_ids"][0]
        item["attention_mask"] = out["attention_mask"][0]
        if self.use_global_attention_mask:
            item["global_attention_mask"] = torch.zeros(len(item["input_ids"]))
            item["global_attention_mask"][0] = 1
        # item["token_type_ids"] = out["token_type_ids"]

        out = self.tokenizer.encode_plus(
            output_text,
            padding="max_length",
            max_length=self.max_output_length,
            truncation=True,
            return_tensors="pt",
        )

        item["labels"] = out["input_ids"][0]
        #item["labels"][item["labels"] == tokenizer.pad_token_id] = -100

        return item

    def __len__(self):
        return len(self.elements)


tokenizer = AutoTokenizer.from_pretrained(args.model_name) # TODO: add special tokens
special_tokens_dict = {'additional_special_tokens': [BS_TOKEN, ES_TOKEN, BT_TOKEN, ET_TOKEN, CONTEXT_TOKEN]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
model.resize_token_embeddings(len(tokenizer))

train_dataset = InternalDataset(
    train_data,
    args.imrad_section,
    args.max_input_length,
    args.max_output_length,
    tokenizer,
    use_global_attention_mask=args.use_global_attention_mask,
)
val_dataset = InternalDataset(
    val_data,
    args.imrad_section,
    args.max_input_length,
    args.max_output_length,
    tokenizer,
    use_global_attention_mask=args.use_global_attention_mask,
)
test_dataset = InternalDataset(
    test_data,
    args.imrad_section,
    args.max_input_length,
    args.max_output_length,
    tokenizer,
    use_global_attention_mask=args.use_global_attention_mask,
)



if args.use_cuda:
    model = model.to("cuda")
else:
    model = model.to("cpu")

model.config.num_beams = 5
model.config.max_length = args.max_output_length
model.config.min_length = 32
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3

rouge = load_metric("rouge")


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    #labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=False)

    label_str = [e.replace(f"{ES_TOKEN}{BS_TOKEN}", "\n\t\t") for e in label_str]
    label_str = [e.replace(f"{BT_TOKEN}", "\n[ *** ] ") for e in label_str]
    label_str = [e.replace(f"{ET_TOKEN}", " [ *** ]\n\t\t") for e in label_str]
    label_str = [e.replace(ES_TOKEN, "") for e in label_str]
    label_str = [e.replace(BS_TOKEN, "") for e in label_str]
    label_str = [e.replace("<pad>", "") for e in label_str]
    label_str = [e.replace("<s>", "") for e in label_str]
    label_str = [e.replace("</s>", "") for e in label_str]

    pred_str = [e.replace(f"{ES_TOKEN}{BS_TOKEN}", "\n\t\t") for e in pred_str]
    pred_str = [e.replace(f"{BT_TOKEN}", "\n[ *** ] ") for e in pred_str]
    pred_str = [e.replace(f"{ET_TOKEN}", " [ *** ]\n\t\t") for e in pred_str]
    pred_str = [e.replace(ES_TOKEN, "") for e in pred_str]
    pred_str = [e.replace(BS_TOKEN, "") for e in pred_str]
    pred_str = [e.replace("<pad>", "") for e in pred_str]
    pred_str = [e.replace("<s>", "") for e in pred_str]
    pred_str = [e.replace("</s>", "") for e in pred_str]

    print ("\n\n\n ********************************")
    for i, e in enumerate(pred_str):
        print("Predicted:", e)
        print ("-------------------")
        print("GT       :", label_str[i])
        print("\n\n\n\n")
    print ("\n\n\n ********************************")

    pred_str  = [e.replace("\n[ *** ] ", "\n") for e in pred_str]
    pred_str  = [e.replace(" [ *** ]\n\t\t", "\n") for e in pred_str]
    pred_str  = [e.replace("\n\t\t", "\n") for e in pred_str]
    label_str = [e.replace("\n[ *** ] ", "\n") for e in label_str]
    label_str = [e.replace(" [ *** ]\n\t\t", "\n") for e in label_str]
    label_str = [e.replace("\n\t\t", "\n") for e in label_str]

    rouge_output1 = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge1"]
    )["rouge1"].mid

    rouge_output2 = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge1_precision": round(rouge_output1.precision, 4),
        "rouge1_recall": round(rouge_output1.recall, 4),
        "rouge1_fmeasure": round(rouge_output1.fmeasure, 4),
        "rouge2_precision": round(rouge_output2.precision, 4),
        "rouge2_recall": round(rouge_output2.recall, 4),
        "rouge2_fmeasure": round(rouge_output2.fmeasure, 4),
    }


# enable fp16 apex training
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    fp16=True,
    output_dir=args.output_dir,
    logging_steps=10,
    save_total_limit=1,
    gradient_accumulation_steps=1,
    num_train_epochs=args.epochs,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    dataloader_num_workers=args.dataloader_num_workers,
    metric_for_best_model="rouge2_fmeasure",
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

trainer.save_model(args.output_dir + "/best_checkpoint/")

"""
python train.py --model_name allenai/PRIMERA-arxiv \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section introduction \
    --output_dir models/result_allenai__PRIMERA-arxiv/ \
    --epochs 10
"""
