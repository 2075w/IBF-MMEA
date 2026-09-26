import json
import os

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# QWen2-VL model configuration
# These parameters control image preprocessing
min_pixels = 256 * 28 * 28  # Minimum pixels for image processing
max_pixels = 1280 * 28 * 28  # Maximum pixels for image processing
model_path = 'Qwen2-VL-7B-Instruct'  # Pretrained model path

# Load the QWen2-VL model with automatic device mapping
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# Load the processor with image size constraints
processor = AutoProcessor.from_pretrained(
    model_path,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)


def read_dict(file_paths):
    """
    Read entity dictionary files and create mappings.

    Args:
        file_paths: List of file paths containing entity ID mappings

    Returns:
        ent2id_dict: List of dictionaries mapping entity names to IDs
        ids: List of sets containing entity IDs
    """
    ent2id_dict = []
    ids = []

    for file_path in file_paths:
        ent2id = {}
        id_set = set()

        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")

                # Process entity names, handling URI format
                if '>' in params[1]:
                    chars = params[1].replace('>', '').split('/')
                    ent = chars[-1]  # Extract the last part as entity name
                else:
                    ent = params[1]

                ent2id[ent] = int(params[0])  # Map entity name to ID
                id_set.add(int(params[0]))  # Add ID to set

        ids.append(id_set)
        ent2id_dict.append(ent2id)

    return ent2id_dict, ids


def load_progress(filepath):
    """
    Load previously processed captions from a file.

    Args:
        filepath: Path to the file containing saved captions

    Returns:
        Dictionary mapping entity IDs to their captions
    """
    print(f"Loading progress from: {filepath}")
    savedata = {}

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            save_data = f.readlines()

        # Parse each line, splitting by tab
        for line in tqdm(save_data):
            try:
                parts = line.split('\t')
                if len(parts) >= 2:
                    entity_id = parts[0]
                    caption = parts[1].strip()
                    savedata[entity_id] = caption
            except Exception as e:
                print(f"Error parsing line: {line}, Error: {e}")

        # Save as JSON for easier inspection
        json_path = filepath.replace("txt", "json")
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(savedata, fp, indent=2)

        return savedata

    return {}


def process_Qwen(ent, img_path):
    """
    Process an image with QWen2-VL model to generate descriptive caption.

    Args:
        ent: Entity ID
        img_path: Path to the image file

    Returns:
        Formatted string: "entity_id\tcaption\n" or None if image not found
    """
    root = "image_path_root"  # Base directory for images
    path = root + img_path

    # Check if the exact file exists, otherwise look for alternatives
    if not os.path.exists(path):
        # Get directory containing the image
        path1 = os.path.dirname(path)

        if os.path.exists(path1):
            # Look for JPG files
            jpg_files = [f for f in os.listdir(path1) if f.lower().endswith('.jpg')]
            if jpg_files:
                # Use the first JPG file found
                path = os.path.join(path1, jpg_files[0])
            else:
                # Look for PNG files
                png_files = [f for f in os.listdir(path1) if f.lower().endswith('.png')]
                if png_files:
                    # Use the first PNG file found
                    path = os.path.join(path1, png_files[0])
                else:
                    # No image files found
                    return None
        else:
            print(f"Path does not exist: {path1}")
            return None

    # Prepare the input message for the vision-language model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "file://" + path,  # File URL for the image
                },
                {
                    "type": "text",
                    "text": "Describe the people, things and places in the photos, mainly the names."
                },
            ],
        }
    ]

    # Process the chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Extract vision information from messages
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare model inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Generate caption with the model
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated tokens to text
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Extract the first sentence and format the output
    caption = output_text[0].split('.')[0] + "."
    return f"{ent}\t{caption}\n"


def multi_process():
    """
    Multi-process image captioning with QWen2-VL model.

    This function processes images to generate captions, maintaining progress
    to allow resuming from where it left off in case of interruption.

    Returns:
        None (writes captions to file)
    """
    # Configuration
    select = "FB15K"  # Dataset name
    select_model = "Qwen"  # Model name
    root_path = "data_path"  # Base data directory

    # Load entity mappings
    ent2id_dict, ids = read_dict([root_path + 'ent_ids_' + str(i) for i in [1, 2]])

    # Define progress file for saving/loading captions
    progress_file = root_path + f"{select_model}_captions_{select}.txt"

    # Get entity to ID mapping for the first dataset
    img_urls = ent2id_dict[0]

    # Load already processed captions
    processed_caption = load_progress(progress_file)

    # Identify remaining entities to process
    remaining_tasks = {
        eid: select + "/google_" + str(eid) + ".jpg"
        for ent, eid in img_urls.items()
        if str(eid) not in processed_caption
    }

    # Process remaining entities
    writedata = ""
    for ent, url in tqdm(remaining_tasks.items(), desc="Generating captions"):
        try:
            result = process_Qwen(ent, url)
            if result:
                writedata += result
        except Exception as e:
            print(f"Error processing entity {ent}: {e}")
            continue

    # Append new captions to file
    if writedata:
        with open(progress_file, "a", encoding="utf-8") as f:
            f.write(writedata)
        print(f"Added {len(writedata.strip().split(chr(10)))} new captions to {progress_file}")
    else:
        print("No new captions generated.")
