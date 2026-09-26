import json
import os
import random
import re

import numpy as np
from tqdm import tqdm
import pandas as pd
from datasets import Dataset

# Instruction template for constructing model input
INSTRUCTION = """
Input:
target entity:
{0}
candidate entities:
{1}
Output:"""

# System prompt defining the entity alignment task rules and format specifications
SYSTEMPROMPT = """Task: Performs entity alignment by determining whether aligned entities exist in candidate entities for a given target entity. When evaluating similarity, focus on information with similar attribute/relationship types, ignore different types of information, and focus on the content in the caption without "without image". Two entities are aligned if the similarity of their type similar information is high. If aligned entity are found, output it's identifiers. If no aligned entity is found, 999999 is outputed. The output should only include alignment results.
Input description:
attribute: entity_id:[(attribute type, attribute values)]
relationship: 152:[('Date of birth', '1999-10-23')] indicates that entity 152 has the 'Date of birth' attribute with the value 'October 23, 1999'.
Example: 152:[('location contains','4')] indicates that entity 152 has four adjacent entities associated with relation type 'location contains'.
caption: Each image description follows the format: {entity_id: "<caption>"}, where <caption> is a natural language sentence, or 'without image' if the entity has no image.
Example: {152: 'A famous historical landmark in Paris'}"""

# Default identifier for unaligned entities
unalign = '999999'


def dict2list(dict_info):
    """
    Convert dictionary key-value pairs to list format.

    Args:
        dict_info: Dictionary in the form {key: value} or [(key, value), ...]

    Returns:
        List containing (key, value) tuples
    """
    list_info = []
    if len(dict_info) == 0:
        return []
    for k, v in dict_info:
        list_info.append((k, v))
    return list_info


def dicts2list(dict_infos):
    """
    Batch convert multiple dictionaries to list format.

    Args:
        dict_infos: List of dictionaries

    Returns:
        2D list where each element is a dictionary converted to list
    """
    list_info = []
    for dict_info in dict_infos:
        list_info.append(dict2list(dict_info))
    return list_info


def load_caption(path, dataset):
    """
    Load image caption information for the specified dataset.

    Args:
        path: Path to data files
        dataset: Dataset name

    Returns:
        Dictionary mapping entity IDs to caption text
    """
    processmodel = "Qwen"
    allcaption = {}

    # Determine which files to load based on dataset type
    if 'icews' in dataset:
        files = ["pic_ent_ids_1.json", "pic_ent_ids_2.json"]
    elif 'FB' in dataset:
        files = ["FB15K.json"]
        if "DB" in path:
            files.append("DB15K.json")
        else:
            files.append("YAGO15K.json")
    elif 'EN' in dataset:
        if "DE" in path:
            files = ["EN_DE_id.json"]
        else:
            files = ["EN_FR_id.json"]

    # Load and merge caption information from all files
    for file in files:
        file = processmodel + "_captions_" + file
        with open(path + file, 'r', encoding='utf-8') as f:
            caption = json.load(f)
            f.close()
        new_cap = {}
        for k, v in caption.items():
            new_cap[int(k)] = v.split('.')[0]  # Take only the first sentence description
        allcaption = {**allcaption, **new_cap}

    # Randomly sample 10% of entities for training/evaluation
    ent_id = list(allcaption.keys())
    random.seed(42)  # Fixed random seed for reproducibility
    random.shuffle(ent_id)
    ent_caption = {}
    for id in ent_id[:int(len(ent_id) * 0.1)]:
        ent_caption[id] = allcaption[id]
    print('Number of entities with captions:', len(ent_caption))
    return ent_caption


def mutlti_instruct(result, start, end, s2t, label="eval", neg=False, pos=False):
    """
    Construct training/evaluation data from candidate entity lists with positive/negative sampling.

    Args:
        result: Query result dictionary, {source_entity: [candidate_entities]}
        start: Starting index of candidate entity list
        end: Ending index of candidate entity list
        s2t: Source to target entity alignment mapping
        label: Dataset type, "train" or "eval"
        neg: Whether to perform negative sampling
        pos: Whether to perform positive sampling

    Returns:
        given_ent: List of source entity IDs
        candi_ents: List of candidate entity ID lists
        incandi: Alignment labels (target entity ID or 999999)
    """
    source_ent_id = []  # Source entity id
    candidate_entities_list = []  # List of candidate entity id lists
    in_candidates = []  # Label (aligned entity or 999999)
    aligned_entity = []  # Target entity id
    no_candi = 0
    no_in_ills = 0

    # Remove duplicate candidate entities while preserving order
    if len(list(result.values())[0]) != len(set(list(result.values())[0])):
        for tar, res in result.items():
            seen = {}  # Track seen elements
            new_res = []  # Store deduplicated results
            for r in res:
                if r not in seen:
                    seen[r] = 1  # First occurrence
                    new_res.append(r)
                else:
                    seen[r] += 1  # Skip duplicates
            result[tar] = new_res

    neg_sum = 0
    pos_sum = 0
    random.seed(42)

    # Process each source entity
    for k, v in result.items():
        if int(k) not in s2t:
            no_in_ills += 1
            continue
        ca = v[start: end]  # Slice candidate list

        # Shuffle candidates for training data to improve generalization
        if label == 'train':
            random.shuffle(ca)

        source_ent_id.append(int(k))
        candidate_entities_list.append(ca)

        # Check if aligned entity exists in candidates
        if s2t[int(k)] in ca:
            in_candidates.append(str(s2t[int(k)]))
            # Add negative sample for training
            if neg and label == 'train':
                neg_sum += 1
                new_ca = ca.copy()
                neg_ent = random.choice(v[end:])  # Negative entity from later candidates
                source_ent_id.append(int(k))
                index = ca.index(s2t[int(k)])
                new_ca[index] = neg_ent  # Replace aligned entity with negative
                candidate_entities_list.append(new_ca)
                in_candidates.append(unalign)
        else:
            in_candidates.append(unalign)

            # Add positive sample for training
            if pos and label == 'train':
                pos_sum += 1
                new_ca = ca.copy()
                source_ent_id.append(int(k))
                neg_ent = random.choice(ca)  # Random candidate to replace
                index = ca.index(neg_ent)
                new_ca[index] = s2t[int(k)]  # Replace with aligned entity
                candidate_entities_list.append(new_ca)
                in_candidates.append(str(s2t[int(k)]))
        aligned_entity.append(s2t[int(k)])

    if neg and label == 'train':
        print("Negative sampling ratio:", neg_sum / len(result))
    if pos and label == 'train':
        print("Positive sampling ratio:", pos_sum / len(result))
    print('Number of entities:', len(source_ent_id))
    print("Sum of alignment entity not in candidate: {0}, number of iteration adding alignment entity: {1}".format(
        no_candi, no_in_ills))

    return source_ent_id, candidate_entities_list, in_candidates


def creat_instruct(file_path, result_path, flag, num, label, dataset, basemodel, il, rate, strage):
    """
    Create instruction data for entity alignment training/evaluation.

    Args:
        file_path: Path to KG data files
        result_path: Path to candidate ranking results
        flag: Return intermediate data if True, else save to file
        num: Number of candidate entities per source entity
        label: Dataset type ("train", "eval", or "reason")
        dataset: Dataset name
        basemodel: Base model name
        il: Incremental learning identifier
        rate: Sampling rate
        strage: Sampling strategy configuration

    Returns:
        If flag=True: source_ent_id, candidate_entities_list, ent_rels, in_candidates
        Else: Saves instruction data to JSON files
    """
    from dataprocess import load_data
    print('result:', result_path)

    # Load candidate ranking results
    with open(result_path, 'r') as f:
        result = json.load(f)

    # Limit to top 100 candidates
    for k, v in result.items():
        result[k] = v[:100]

    # Parse strategy parameters
    fliter = 'matf' in strage or 'nodec' in strage or 'one' == strage
    negpos = 'np' in strage and label == 'Train'
    pos = 'p' in strage
    print('Strategy parameters: Attribute filtering: {0}, Negative sampling: {1}, Positive sampling: {2}'.format(fliter,
                                                                                                                 negpos,
                                                                                                                 pos))

    # Load KG data
    KGs, _, _, _ = load_data(file_path, dataset, 0.2, attr_flag=True, save=False, llm_flage=True, fliter=fliter)

    # Load captions
    if os.path.exists(file_path + dataset):
        caption = load_caption(file_path + dataset + '/', dataset)
    else:
        if 'EN' in dataset:
            caption = load_caption(file_path + 'imgs/', [dataset.split('_15K')[0]])
        elif 'icews' in dataset:
            caption = load_caption(file_path + 'imgs/', [dataset.split('_15K')[0]])
        else:
            print('Caption directory not found')
            caption = [{}, {}]

    # Extract KG components
    entities = KGs["node"]  # id2attr
    old_ent_rels = KGs["rel"]  # id2rel
    ills = KGs["ills"]  # alignment pairs: [(sid, tid)]
    del KGs

    # Transform relationship data to list format
    ent_rels = {}
    for ent, rels in old_ent_rels.items():
        for rel, neigs in rels.items():
            if ent not in ent_rels:
                ent_rels[ent] = [(rel, str(len(neigs)))]
            else:
                ent_rels[ent].append((rel, str(len(neigs))))

    # Create source-target and target-source mappings
    s2t = {}
    t2s = {}
    for i, j in ills:
        s2t[i] = j
        t2s[j] = i

    # Regex pattern to extract entity IDs from instruction
    candi_ids_part = r'\d+(?=\:)'

    # Determine number of data slices (rounds)
    rounds = 2 if label == 'train' else 5
    if strage == 'one' or strage == 'none':
        rounds = 1
        num = 20

    start = 0
    end = num
    save_paths = []

    # Generate initial data slice
    source_ent_id, candidate_entities_list, in_candidates = mutlti_instruct(result, start, end, s2t, label, neg=negpos)

    if flag:
        return source_ent_id, candidate_entities_list, ent_rels, in_candidates
    else:
        # Process each data slice
        for number in range(rounds):
            print("start:", start, "end:", end)
            error = 0
            instructs = []

            # Create instruction for each entity
            for i in tqdm(range(len(source_ent_id))):
                attr = {}
                rel = {}
                cap = {}

                # Get target entity caption
                ent_caption = caption.get(source_ent_id[i], "without image")

                # Get target entity attributes
                try:
                    tar_atttrs = list(entities[source_ent_id[i]]["attrs"])
                except:
                    tar_atttrs = []

                # Construct target entity info
                info = {"attribute": tar_atttrs,
                        "relationship": ent_rels.get(source_ent_id[i], []),
                        "caption": ent_caption}

                # Get candidate entity information
                for index, j in enumerate(candidate_entities_list[i]):
                    try:
                        cand_atttrs = list(entities[j]["attrs"])
                    except:
                        cand_atttrs = []
                    attr[j] = cand_atttrs
                    rel[j] = ent_rels.get(j, [])
                    cap[j] = caption.get(j, "without image")

                candi = {"attribute": attr, "relationship": rel, "caption": cap}

                # Construct full instruction
                instruct = {"instruction": SYSTEMPROMPT + INSTRUCTION.format(info, candi),
                            "input": "",
                            "output": str(in_candidates[i]),
                            "target": str(source_ent_id[i]),
                            }

                # Validate that output is in candidate list
                candis = list(set(re.findall(candi_ids_part,
                                             instruct['instruction'].split("candidate entities:")[-1].split(
                                                 'relationships')[0])))
                if str(instruct['output']) not in candis and str(instruct['output']) != unalign:
                    print(instruct['output'], candis)
                    error += 1

                instructs.append(instruct)

            print('Number of erroneous data points:', error, 'Total data points:', len(source_ent_id))

            # Save instruction data
            save_dir = "./result/{0}/{1}/{2}_{3}_{4}".format(basemodel, il,
                                                             dataset if 'icews' in dataset else dataset.replace('/norm',
                                                                                                                '')[
                                                                                                :-3],
                                                             rate, strage)
            save_path = save_dir + "/instruct_zero_{0}-{1}_{2}.json".format(start, end, label)
            save_paths.append(save_path)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            print(save_path)
            with open(save_path, 'w', encoding="utf-8") as f:
                json.dump(instructs, f, ensure_ascii=False)
                f.close()

            if number >= rounds - 1:
                break

            # Prepare next slice
            start = end
            end += num
            if label != 'train':
                source_ent_id, candidate_entities_list, in_candidates = mutlti_instruct(result, start, end, s2t)
            else:
                source_ent_id, candidate_entities_list, in_candidates = mutlti_instruct(result, start, end, s2t, label,
                                                                                        pos=negpos)


def input_static(basemodel, il, ds, rate, strage, datanum, label):
    """
    Analyze statistics of generated instruction data.

    Args:
        basemodel: Base model name
        il: Incremental learning identifier
        ds: Dataset identifier
        rate: Sampling rate
        strage: Sampling strategy
        datanum: Number of candidates per entity
        label: Dataset type

    Prints:
        Average input length, median input length, and accuracy
    """
    candi_ids_part = r'\d+(?=\:)'
    rounds = 2 if label == 'train' else 3
    if strage == 'one' or strage == 'none':
        rounds = 1
        datanum = 20

    start = 0
    end = datanum
    data_paths = []

    # Construct paths to all data slices
    for i in range(rounds):
        data_paths.append(
            "./result/{0}/{1}/{2}_{3}_{4}/instruct_zero_{5}-{6}_{7}.json".format(basemodel, il, ds, rate, strage,
                                                                                 start, end, label))
        start = end
        end += 10

    print(data_paths)
    sums = []
    num = 0
    acc = 0
    dfs = []

    # Load and analyze each data slice
    for path in data_paths:
        df = pd.read_json(path)
        ds = Dataset.from_pandas(df)
        dfs.append(ds)

    for df in dfs:
        for d in df:
            ins, _, output, target = list(d.values())
            # Extract candidate IDs from instruction
            candis = list(
                set(re.findall(candi_ids_part, ins.split("candidate entities:")[-1].split('relationships')[0])))
            if str(output) in candis:
                acc += 1
            sums.append(len(ins))
            num += 1

    print('Average input length:', np.mean(sums), 'Median input length:', np.median(sums))
    print('Accuracy:', acc * rounds / num)


if __name__ == "__main__":
    # Main execution block
    basemodel = "based-Embedding model"
    il = "nil"
    split = ''
    root = 'data_root'

    # Process multiple datasets
    for dataset in ["FBYG", "FBDB"]:
        rate_i = 1
        print("basemodel is:", basemodel, "dataset is:", dataset, "il:", il)

        # Determine dataset paths and naming
        if 'FB' in dataset:
            root += 'mmkb'
            dataset = dataset + '15K'
            dataset1 = dataset
        elif 'icews' in dataset:
            root += ''
            dataset = dataset
            dataset1 = dataset
        else:
            root += 'OpenEA'
            dataset1 = dataset + '_15K_V2_norm'
            dataset = dataset + '_15K_V2/norm'

        dataset1 = basemodel + "_" + dataset1

        rate = '0.2'

        # Generate training and evaluation data
        for label in ["train", "eval"]:
            if label == "train":
                strage = "matf_l_np"  # Training with negative sampling
            else:
                strage = "matf"  # Evaluation without sampling

            if label == "reason":
                label1 = "eval"
            else:
                label1 = label

            datanum = 10

            # Create instruction data
            creat_instruct(file_path=root,
                           result_path="./result/{0}/{1}/{2}_{3}_{1}_left_{4}.json".format(
                               basemodel, il, dataset1 if '\'' not in dataset1 else dataset1[5:], rate, label1),
                           flag=0, num=datanum, label=label, dataset=dataset,
                           basemodel=basemodel, il=il, rate=rate, strage=strage)

            # Analyze generated data statistics
            input_static(basemodel, il, dataset.replace('/norm', '')[:-3], rate, strage, datanum, label)
