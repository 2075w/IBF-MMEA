# IBF-MMEA

# Dataset Preparation Guide

##  MMKB
The MMKB dataset comes from the [EVA repository](https://github.com/mniepert/mmkb).

##  Multi-OpenEA
The Multi-OpenEA dataset comes from the [OpenEA repository](https://github.com/nju-websoft/OpenEA).

# How to Run

## Step 1: Obtain Candidate Entity Ranking
Generate an initial list of potential matching entities from the target Knowledge Graph (KG) for each entity in the source KG.
as: entity_id: [cadidate_entity_1, cadidate_entity_2, ...]

## Step 2: Attribute Filtering & Prompt
Obtain the caption of the entity image.

```bash
python get_caption.py
```
Prepare structured prompts for the LLM by incorporating entity attributes, using the candidate list from Step 1.

```bash
python creat_prompt.py
```

## Step 3: Fine-tuning & Inference

```bash
# Training
accelerate launch llm_api.py
  --lora
  --neg_sample 2
  --dropout 0.2
  --epoch 5 
  --strategy matf_l_np
  --dataset FBYG

# inferencing
python llm_api.py
--test
--neg_sample 2
--epoch 5
--strategy matf_l_np
--dataset FBYG
```

The main Python script file containing model definition, training, and testing logic.

--lora​ - Enable LoRA fine-tuning (training only)

--test​ - Switch to testing mode

--epoch N​ - Training: N epochs; Testing: load epoch N checkpoint

--strage NAME​ - Strategy identifier (must match between train/test)

--dataset NAME​ - Dataset identifier

--neg_sample N​ - Negative samples (default: 2)

--dropout RATE​ - Dropout rate 0.0-1.0 (default: 0.2)

# Prompt
```bash
Task: Performs entity alignment by determining whether aligned entities exist in candidate entities for a given target entity. When evaluating similarity, focus on information with similar attribute/relationship types, ignore different types of information, and focus on the content in the caption without "without image". Two entities are aligned if the similarity of their type similar information is high. If aligned entity are found, output it's identifiers. If no aligned entity is found, 999999 is outputed. The output should only include alignment results.
Input description:
attribute: entity_id:[(attribute type, attribute values)]
relationship: 152:[('Date of birth', '1999-10-23')] indicates that entity 152 has the 'Date of birth' attribute with the value 'October 23, 1999'.
Example: 152:[('location contains','4')] indicates that entity 152 has four adjacent entities associated with relation type 'location contains'.
caption: Each image description follows the format: {entity_id: "<caption>"}, where <caption> is a natural language sentence, or 'without image' if the entity has no image.
Example: {152: 'A famous historical landmark in Paris'}
```
