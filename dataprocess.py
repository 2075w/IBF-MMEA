import os
import pickle
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def date2float(date):
    """
    Convert a date string to a float representation.

    Args:
        date: Date string in "YYYY-MM-DD" format

    Returns:
        String representing the year. For December dates, returns next year.
    """
    if re.match(r'\d+-\d+-\d+', date):
        year = date.split('-')[0]
        month = date.split('-')[1]
        if month == '12':
            year = str(int(year) + 1)
        return year
    elif re.match(r'-\d+-\d+-\d+', date):
        year = date.split('-')[1]
        return year
    else:
        return date


class InfoFilter:
    """
    Information filter for entity attributes based on cosine similarity and IDF.

    Attributes with high TF values (e.g., 0.2) for attribute types and 
    lower values (e.g., 0.1) for relation types are prioritized.
    """

    def __init__(self, id2attr, attr_count, sent_num, allent):
        """
        Initialize the information filter.

        Args:
            id2attr: List of dictionaries mapping attribute IDs to attribute names for source and target
            attr_count: List of dictionaries with attribute frequencies for source and target
            sent_num: Number of source entities
            allent: Total number of entities (source + target)
        """
        # Initialize sentence transformer model for embedding generation
        self.model = SentenceTransformer('model_path').to('cuda:0')
        self.id2attr = id2attr
        # Create reverse mapping from attribute names to IDs
        self.attr2id = [{v: k for k, v in i2a.items()} for i2a in id2attr]
        self.slen = [0, len(id2attr[0])]
        self.attr_count = attr_count
        self.ent_num = [sent_num, allent - sent_num]
        self.score = None
        self.score_st = None
        # Pre-compute cosine similarities
        self.cos_score()

    def cos_score(self):
        """Calculate cosine similarity between source and target attribute embeddings."""
        # Extract attribute names from both source and target
        s_infos = [infos for infos in self.id2attr[0].values()]
        t_infos = [infos for infos in self.id2attr[1].values()]
        infos = s_infos + t_infos
        source_len = len(self.id2attr[0])

        # Generate embeddings in batches to manage memory
        infos_embeddings = []
        batch_size = 128
        for i in tqdm(range(0, len(infos), batch_size)):
            key_sents = infos[i:i + batch_size]
            infos_embeddings.append(self.model.encode(key_sents))
        infos_embeddings = np.concatenate(infos_embeddings, axis=0)

        # Calculate cosine similarity matrix
        self.score = cosine_similarity(infos_embeddings[:source_len], infos_embeddings[source_len:])
        # Store max similarities for source→target and target→source
        self.score_st = [np.max(self.score, axis=1), np.max(self.score, axis=0)]

    def KG_fliter_sim_idf(self, entities):
        """
        Filter entity attributes using similarity and IDF weighting.

        Args:
            entities: Dictionary mapping entity IDs to their attributes

        Returns:
            Dictionary with filtered attributes for each entity
        """
        new_entities = {}
        tar_max = np.max(self.score, axis=0)
        sou_max = np.max(self.score, axis=1)
        A_score = np.concatenate([sou_max, tar_max])

        # Initialize dictionaries for attribute scoring
        type_score = {}
        type_sim = {}
        type_fre = {}
        type_idf = {}

        # Calculate similarity, frequency, and IDF scores for each attribute
        for i, id2attr in enumerate(self.id2attr):
            for id, attr in id2attr.items():
                if attr in type_sim:
                    type_sim[attr] = 1.0
                else:
                    type_sim[attr] = A_score[id]
                    type_fre[attr] = self.attr_count[i][attr] / self.ent_num[i]
                    type_idf[attr] = type_sim[attr] * type_fre[attr]

        # Calculate IDF threshold (mean of all IDF scores)
        idf_thre = np.mean(list(type_idf.values()))

        # Select attributes with IDF above threshold
        count = 0
        save_attrs = set()
        for attr, score in type_idf.items():
            if score > idf_thre:
                save_attrs.add(attr)
                count += 1
        print(f"Selected {count} attributes out of {len(type_score)}")

        # Filter entity attributes
        rate = 0
        all_attr = 0
        fliter_attr = 0
        for id, attrs in entities.items():
            new_attrs = set()
            all_attr += len(attrs)
            if len(attrs) == 0:
                continue
            for attr, value in attrs:
                # Keep attribute if in saved set or if entity has very few attributes
                if attr in save_attrs or 1 / len(attrs) > 0.5:
                    new_attrs.add((attr, value))
            new_entities[id] = list(new_attrs)
            rate += len(new_attrs) / len(attrs)

        # Statistics
        count = 0
        for id, attrs in new_entities.items():
            fliter_attr += len(attrs)
            if len(attrs) > 0:
                count += 1

        print(f'Average attribute filtering ratio per entity: {round(rate / len(entities), 4)}')
        print(f'Entities with attributes after filtering: {count}')
        print(f'Overall attribute filtering ratio: {round(fliter_attr / all_attr, 4)}')
        return new_entities

    def save(self, entities, paths):
        """
        Save filtered entity-attribute relations to files.

        Args:
            entities: Dictionary of entities with their filtered attributes
            paths: List of two file paths for source and target entities
        """
        save_attr0 = ""
        save_attr1 = ""

        # Format entities for saving (tab-separated: entity_id, attribute, value)
        for k, v in entities.items():
            if k < self.sent_num:  # Source entities
                if len(v) == 0:
                    continue
                for attr, val in v:
                    save_attr0 += f'{k}\t{attr}\t{val}\n'
            else:  # Target entities
                if len(v) == 0:
                    continue
                for attr, val in v:
                    save_attr1 += f'{k}\t{attr}\t{val}\n'

        # Save to files
        with open(paths[0].replace('attr', 'flit_attr'), 'w', encoding='utf-8') as f:
            f.write(save_attr0)
        with open(paths[1].replace('attr', 'flit_attr'), 'w', encoding='utf-8') as f:
            f.write(save_attr1)


def find_neigh(ill, triples):
    """
    Find neighboring entities for a set of entities in the knowledge graph.

    Args:
        ill: Set of entity IDs to find neighbors for
        triples: List of (head, relation, tail) triples

    Returns:
        head: Dictionary mapping entity to set of (relation, head) where entity is tail
        tail: Dictionary mapping entity to set of (relation, tail) where entity is head
    """
    head = {}
    tail = {}
    for h, r, t in triples:
        if h in ill:
            if h in tail:
                tail[h].add((r, t))
            else:
                tail[h] = set([(r, t)])
        if t in ill:
            if t in head:
                head[t].add((r, h))
            else:
                head[t] = set([(r, h)])
    return head, tail


def create_graph(ids, id2attrs, triples, rid2rel, id2attr=None, id2ent_dicts=None):
    """
    Create graph representation from knowledge graph data.

    Args:
        ids: [source_ids, target_ids] - sets of entity IDs
        id2attrs: Dictionary mapping entity ID to list of (attribute, value) pairs
        triples: List of (head, relation, tail) triples
        rid2rel: Dictionary mapping relation ID to relation name
        id2attr: Dictionary mapping attribute ID to attribute value
        id2ent_dicts: List of dictionaries mapping entity ID to entity name

    Returns:
        node: Dictionary with entity structure including attributes and neighbors
        relation: Dictionary with relation information
    """
    node = {}
    relation = {}
    relation_2hop = {}

    # Find 1-hop neighbors for source and target entities
    s_h, s_t = find_neigh(ids[0], triples)
    t_h, t_t = find_neigh(ids[1], triples)

    # Initialize node structures
    for id in ids[0].union(ids[1]):
        node[id] = {}
        if id2attrs:
            try:
                if id2attr:
                    node[id]["attrs"] = id2attr[id] if len(id2attr[id]) > 0 else []
                else:
                    attrs = []
                    for attr, val in id2attrs[id]:
                        if attr in id2attr.values():
                            attrs.append((attr, val))
                    node[id]["attrs"] = attrs
            except:
                node[id]["attrs"] = []
            if type(node[id]["attrs"]) is set:
                node[id]["attrs"] = list(node[id]["attrs"])

    # Add entity names if available
    if id2ent_dicts:
        id2ent = {**id2ent_dicts[0], **id2ent_dicts[1]}
        for id in ids[0].union(ids[1]):
            node[id]["name"] = id2ent[id]

    # Build relation dictionaries
    for ent in node.keys():
        relation[ent] = {}
        relation_2hop[ent] = {}

        # Get head and tail neighbors
        if ent in ids[0]:  # Source entity
            h = set(s_h[ent] if ent in s_h else [])
            t = set(s_t[ent] if ent in s_t else [])
        else:  # Target entity
            h = set(t_h[ent] if ent in t_h else [])
            t = set(t_t[ent] if ent in t_t else [])

        neighs = h.union(t)

        # Add relations to the dictionary
        for r, n_id in neighs:
            if r in rid2rel:
                if rid2rel[r] not in relation[ent]:
                    relation[ent][rid2rel[r]] = {n_id}
                else:
                    relation[ent][rid2rel[r]].add(n_id)

    return node, relation


def load_img(ent2id, path):
    """Load image embeddings from pickle file, handling missing embeddings."""
    print("Loading image embeddings...")
    e_num = max(ent2id[1].values()) + 1

    # Load pre-computed image embeddings
    img_dict = pickle.load(open(path, "rb"))
    imgs_np = np.array(list(img_dict.values()))

    # Calculate statistics for random initialization of missing embeddings
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)

    # Create embedding matrix, filling missing entities with random normal distribution
    img_embd = np.array([img_dict[i] if i in img_dict
                         else np.random.normal(mean, std, mean.shape[0])
                         for i in range(e_num)])

    # Alternative: fill missing embeddings with zeros
    # img_embd = np.array([img_dict[i] if i in img_dict 
    #                      else np.zeros_like(img_dict[0])
    #                      for i in range(e_num)])

    # Track entities without image embeddings
    nv_ent = set()
    for i in range(e_num):
        if i not in img_dict:
            nv_ent.add(i)

    # Normalize embeddings
    min_emb = np.mean(img_embd, axis=0)
    max_emb = np.max(img_embd, axis=0)
    img_embd = (img_embd - min_emb) / (max_emb - min_emb + 1e-7)
    img_embd = np.around(img_embd, decimals=6)

    return img_embd


def load_attr(fns, ids_list):
    """
    Load and process attribute data from files.

    Args:
        fns: List of attribute file paths
        ids_list: List of entity ID sets for source and target

    Returns:
        id2attrs: Dictionary mapping entity ID to its attributes
        attr: List of attribute ID to name mappings
        attr_count: List of dictionaries with attribute frequencies
    """
    import wordsegment as ws
    ws.load()

    attr = [{}, {}]
    attr_count = [{}, {}]
    a_attr = {}
    t_attr = {}
    atts = [a_attr, t_attr]
    id2attrs = {}
    index = 0
    a_id = 0

    for fn in fns:
        ent2id = ids_list[index]
        for id in ent2id:
            id2attrs[id] = []

        with open(fn, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        attrs = []
        for line in tqdm(set(lines)):
            th = line.split('\t')
            e_id = int(th[0])
            a1 = th[1].split('/')[-1].strip('>').replace('_', '').split('.')[-1].split('#')[-1]

            # Word segmentation for attribute names without spaces
            if ' ' not in a1:
                a = ''
                for w in ws.segment(a1):
                    a += w + ' '
            else:
                a = a1
            a = a.strip()
            attrs.append(a)

            if e_id not in ent2id:
                continue

            # Extract attribute value
            try:
                value = th[2].split("\"")[1]
            except:
                value = th[2]

            # Skip URL values
            if 'http' in value:
                continue

            value = value.strip()
            value = date2float(value)

            # Update attribute dictionaries
            if a not in attr[index].values():
                attr[index][a_id] = a
                attr_count[index][a] = [e_id]
                atts[index][a] = 1
                a_id += 1
            else:
                if e_id not in attr_count[index][a]:
                    attr_count[index][a].append(e_id)
                atts[index][a] += 1

            id2attrs[e_id].append((a, value))

        print(f"Attribute type count for {fn}: {len(set(attrs))}")
        index += 1

    # Convert to sets for deduplication
    for k, v in id2attrs.items():
        id2attrs[k] = set(v)

    # Convert entity lists to counts
    for i in range(len(attr_count)):
        for a, eids in attr_count[i].items():
            attr_count[i][a] = len(eids)

    return id2attrs, attr, attr_count


def read_file(file_paths):
    """Read triples from files and return as list of tuples."""
    tups = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")
                tups.append(tuple([int(x) for x in params[:3]]))
    return tups


def read_dict(file_paths):
    """
    Read entity dictionaries from files.

    Returns:
        ent2id_dict: List of entity name to ID mappings
        ids: List of entity ID sets
        id2ent_dict: List of ID to entity name mappings
    """
    ent2id_dict = []
    id2ent_dict = []
    ids = []

    for file_path in file_paths:
        ent2id = {}
        id2ent = {}
        id_set = set()

        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")
                if '>' in params[1]:
                    chars = params[1].replace('>', '').split('resource/')[-1]
                    ent = chars.replace('/', '_')
                    if '/' in chars:
                        ent2id[chars.split('/')[-1]] = int(params[0])
                else:
                    ent = params[1]

                if ent in ent2id:
                    print(f"Duplicate entity: {ent}, previous ID: {id2ent[ent2id[ent]]}")

                ent2id[ent] = int(params[0])
                id2ent[int(params[0])] = ent
                id_set.add(int(params[0]))

        ids.append(id_set)
        ent2id_dict.append(ent2id)
        id2ent_dict.append(id2ent)

    return ent2id_dict, ids, id2ent_dict


def read_triples(file_paths, ent2ids):
    """
    Read triples and relation information from files.

    Returns:
        id2rel: List of relation ID to name mappings
        triples: List of (head, relation_id, tail) tuples
        rel1: Number of relations in first file
        rel_count: List of dictionaries with relation frequencies
    """
    id2rel = [{}, {}]
    rels = []
    triples = []
    index = 0
    tri1 = 0
    rel1 = 0
    id = 0
    rel2id = {}
    rel_count = [{}, {}]

    for file_path in file_paths:
        reltype = []
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")
                rel = params[1]

                # Process relation names
                if "/" in rel:
                    words = rel.split("/")
                    if len(words) > 2:
                        rel = words[-2] + " " + words[-1]
                    else:
                        rel = rel.replace('/', ' ').replace("#", ' ')

                # Update relation dictionaries
                if rel not in rel_count[index]:
                    id2rel[index][id] = rel
                    rel2id[rel] = id
                    rel_count[index][rel] = 1
                    id += 1
                else:
                    rel_count[index][rel] += 1

                rels.append(rel)
                reltype.append(rel)

                # Convert entity names to IDs
                try:
                    triples.append((ent2ids[index % 2][params[0]], rel, ent2ids[index % 2][params[2]]))
                except:
                    print("Error processing triple")

        print(f"Relation type count for {file_path}: {len(set(reltype))}, total relations: {len(reltype)}")

        if index % 2 == 0:
            tri1 = len(rels) - 1
            rel1 = len(rel2id)
        index += 1

    # Convert relation names to IDs in triples
    for i in range(len(triples)):
        triples[i] = (triples[i][0], rel2id[triples[i][1]], triples[i][2])

    return id2rel, triples, rel1, rel_count


def load_data(file_path, dirname, data_rate, attr_flag=False, img_flag=False,
              save=False, llm_flage=False, fliter=False):
    """
    Main data loading function for knowledge graph datasets.

    Args:
        file_path: Base directory path
        dirname: Dataset directory name
        data_rate: Proportion of data to use for training
        attr_flag: Whether to load attributes
        img_flag: Whether to load image embeddings
        save: Whether to save processed data
        llm_flage: Whether to create graph structure for LLM
        fliter: Whether to filter attributes

    Returns:
        KGs: Dictionary containing all knowledge graph data
        source_non_train: Source entities not in training
        target_non_train: Target entities not in training
        train_ill: Training alignment pairs
        test_ill: Testing alignment pairs
    """
    print("Loading data...")
    rel_fliter = False
    file_dir = file_path + dirname

    # Determine image embedding path based on dataset
    if "icews" in dirname:
        img_path = file_dir + f"/images/{dirname}_id_img_feature_dict.pkl"
    elif 'DB' in dirname:
        img_path = file_dir + "/FB15K_DB15K_id_img_feature_dict.pkl"
    elif 'YG' in dirname:
        img_path = file_dir + "/FB15K_YAGO15K_id_img_feature_dict.pkl"
    elif 'DBP' in dirname:
        img_path = file_path + f"pkls/{dirname}_id_img_feature_dict.pkl"
    else:
        img_path = file_path + f"pkls/{dirname.split('/')[0]}_id_img_feature_dict.pkl"

    print(f'Image embedding path: {img_path}')

    KGs = {}
    ent2id_dict, ids, id2ent_dicts = read_dict([file_dir + "/ent_ids_" + str(i) for i in [1, 2]])
    ENT_NUM = len(ids[0])
    KGs['ent_num'] = ENT_NUM

    # Load triples and relations
    if 'icews' in file_dir:
        _, _, r_id2rel = read_dict([file_dir + "/ent_ids_" + str(i) for i in [1, 2]])
        rel1 = len(r_id2rel[0])
        triples = read_file([file_dir + "/triples_" + str(i) for i in [1, 2]])
        rel_count = [{}, {}]

        # Count relation frequencies
        for tri in triples:
            index = 0
            h, r, t = tri
            if r > rel1:
                index = 1
            if r not in rel_count[index]:
                rel_count[index][r_id2rel[index][r]] = 1
            else:
                rel_count[index][r_id2rel[index][r]] += 1
    else:
        r_id2rel, triples, rel1, rel_count = read_triples(
            [file_dir + "/rel_triples_" + str(i) for i in [1, 2]], ent2id_dict
        )

    KGs['rel_count'] = rel_count

    # Remove self-loops
    for tri in triples:
        if tri[0] == tri[2]:
            triples.remove(tri)

    KGs['tri_num'] = len(triples)
    KGs['ids'] = ids

    # Load entity alignment pairs
    if 'icews' not in file_dir:
        ills = read_file([file_dir + "/ill_ent_ids"])
        KGs['ills'] = ills
        train_ill = np.array(ills[:int(len(ills) // 1 * data_rate)])
        test_ill = np.array(ills[int(len(ills) // 1 * data_rate):])
    else:
        ills = None
        train_ill = np.array(read_file([file_dir + "/ref_pairs"]))
        test_ill = np.array(read_file([file_dir + "/sup_pairs"]))
        KGs['ills'] = np.concatenate([train_ill, test_ill], axis=0)

    # Get entities not in training
    source_non_train = list(set(ids[0]) - set(train_ill[:, 0]))
    target_non_train = list(set(ids[1]) - set(train_ill[:, 1]))

    # Load attributes if requested
    if attr_flag and ills:
        a1 = os.path.join(file_dir, 'attr_triples_1')
        a2 = os.path.join(file_dir, 'attr_triples_2')
        print('Processing attributes...')
        id2attrs, attrs, attr_count = load_attr([a1, a2], ids)
        KGs['attr_count'] = attr_count
        KGs['id2attr'] = attrs

        # Filter attributes if requested
        if fliter:
            attr_fliter = InfoFilter(attrs, KGs['attr_count'], KGs['ent_num'], max(ids[1]) + 1)
            new_id2attr = attr_fliter.KG_fliter_sim_idf(id2attrs)
        else:
            new_id2attr = id2attrs
    else:
        id2attrs = None
        new_id2attr = None

    # Create graph structure for LLM if requested
    if llm_flage:
        KGs["node"], KGs["rel"] = create_graph(
            ids, id2attrs, triples,
            {**r_id2rel[0], **r_id2rel[1]},
            new_id2attr,
            id2ent_dicts
        )
        KGs["ent2id"] = ent2id_dict

    # Clean up to save memory
    del ids, r_id2rel, id2attrs, triples

    # Load image embeddings if requested
    if img_flag:
        KGs["images_list"] = load_img(ent2id_dict, img_path)

    return KGs, [source_non_train, target_non_train], train_ill, test_ill
