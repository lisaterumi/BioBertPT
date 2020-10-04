** **This is a work in progress** **

<img src="./logo-biobertpr1.png" alt="Logo BioBERTpt">

# BioBERTpt - Portuguese  Clinical BERT

This repository contains fine-tuned [BERT](https://github.com/google-research/bert) models trained on the clinical domain for Portuguese language. Pre-trained BERT-multilingual-cased were fine-tuned with clinical narratives from Brazilian hospitals and abstracts of scientific papers from Pubmed and Scielo.

## NER Experiment in SemClinBr Corpora

We evaluate our models on SemClinBr, a semantically annotated corpus for Portuguese clinical NER, containing 1,000 labeled clinical notes. These corpus comprehended 100 UMLS semantic types, summarized in 13 groups of entities: Disorders, Chemicals and Drugs, Medical Procedure, Diagnostic Procedure, Disease Or Syndrome, Findings, Health Care Activity, Laboratory or Test Result, Medical Device, Pharmacologic Substance, Quantitative Concept, Sign or Symptom and Therapeutic or Preventive Procedure.

Complete F1-score results are:

| Entity / Model | Disorders | ChemicalDrugs | Procedures | DiagProced | DiseaseSynd | Findings | Heatlh | Laboratory | Medical | Pharmacologic | Quantitative | Sign | Therapeutic |
|BERT multilingual uncased|0.787|------|------|------|------|------|------|------|------|------|------|------|------|
|BERT multilingual cased|0.782|------|------|------|------|------|------|------|------|------|------|------|------|
|Portuguese BERT large|0.625|------|------|------|------|------|------|------|------|------|------|------|------|
|Portuguese BERT base|0.784|------|------|------|------|------|------|------|------|------|------|------|------|
|BioBERtpt(bio)|0.785|------|------|------|------|------|------|------|------|------|------|------|------|
|BioBERtpt(clin)|0.781|------|------|------|------|------|------|------|------|------|------|------|------|
|BioBERtpt(all)|0.791|------|------|------|------|------|------|------|------|------|------|------|------|

   

## Download BioBERTpt

| Model | Domain | PyTorch checkpoint | 
|------|-------|:-------------------------:|
|`BioBERTpt (all)`  | Clinical + Biomedical |  [Download](https://drive.google.com/open?id=1PrGzj7B0B6rXjPmKoFFOXa1gGjVVHuwA) |
|`BioBERTpt (clin)`  | Clinical | [Download](https://drive.google.com/open?id=1GIOqxPMxeW8sc4EyQ8s1ol3RFWgsBFte) |
|`BioBERTpt (bio)`  | Biomedical | [Download](https://drive.google.com/open?id=16D0WA1QMoycvA0tR3KyVdMU1-vpw98sp) |

## Prerequisite
-----
Please download the amazin [Huggingface implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

## Usage
-----
1. Download the BioBERTpt (we recomend **BioBERTpt(all)**, see above) and unzip it into <bert_directory>.

2. Install the environment necessary to HuggingFace. 

3. For a NER task, for example, you just need to load the model.

```
model = BertForTokenClassification.from_pretrained(<bert_directory>)
```

For more information, you can refer to these [examples](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples).

## Reproduce BioBERTpt
-----

To replicate our work, or fine-tune you own model, just do this steps:

```
git clone https://github.com/huggingface/transformers
cd transformers
pip install .

mkdir data

# please put your corpus file in this folder in a txt format

python examples/run_language_modeling.py --output_dir=output --model_type=bert \
    --model_name_or_path=bert-base-multilingual-cased --do_train --train_data_file=data/corpus.txt  --num_train_epochs 15 --mlm \
	--learning_rate 1e-5  --per_gpu_train_batch_size 16 --seed 666 --block_size=512
```

## Citation

(soon)
