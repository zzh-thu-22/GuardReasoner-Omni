import os
import gc
import re
import json
import argparse
from template import INSTRUCTION
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser(description="GuardReasoner-Omni Hard Sample Mining")
parser.add_argument("--model_path", type=str, default="your_model_path", help="model path")
args = parser.parse_args()

MODEL_NAME = f"{args.model_path.split('/')[-1]}"
OUTPUT_FILE = f"hard_sample_{MODEL_NAME}.jsonl"


vllm_model = LLM(model=args.model_path, gpu_memory_utilization=0.9, max_num_seqs=256, max_model_len=32768, limit_mm_per_prompt={"image": 10, "video": 10})
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=4096, n=4)
processor = AutoProcessor.from_pretrained(args.model_path)



def extract_labels(text):
    """ Extract labels from the text. """
    try:
        request_label = re.search(r'Request:\s*(harmful|unharmful|none|harmless)', text, re.IGNORECASE).groups()[0].lower()
    except:
        request_label = None

    try:
        response_label = re.search(r'Response:\s*(harmful|unharmful|none|harmless)', text, re.IGNORECASE).groups()[0].lower()
    except:
        response_label = None
    
    if request_label in ["none", "harmless"]:
        request_label = "unharmful"
    if response_label in ["none", "harmless"]:
        response_label = "unharmful"
    return request_label, response_label


BATCH_SIZE = 500

all_data = []
with open(f"./train/LLaMA-Factory/data/GuardReasoner-OmniTrain.json", encoding="utf-8") as file:
    all_data = json.load(file)

print(f"Total data loaded: {len(all_data)}")


for i in range(0, len(all_data), BATCH_SIZE):
    batch_data = all_data[i : i + BATCH_SIZE]
    print(f"Processing batch {i} to {i + len(batch_data)}...")

    input_list = []
    metadata = []

    
    for sample in batch_data:
        save_dict = {}
        try: 
            label, label_res = extract_labels(sample["messages"][2]["content"])
            
          
            if len(sample.get("videos", [])) == 1:
                video_path = sample["videos"][0]
                messages = [
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": [
                        {"type": "video", "video": video_path, "max_frames":128, "max_pixels":128*32*32},
                        {"type": "text", "text": sample["messages"][1]["content"]},
                    ]},
                ]
                
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=processor.image_processor.patch_size,
                    return_video_kwargs=True,
                    return_video_metadata=True
                )
                
                mm_data = {}
                if image_inputs is not None: mm_data["image"] = image_inputs
                if video_inputs is not None: mm_data["video"] = video_inputs
                
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt = prompt.replace("<|vision_start|><|video_pad|><|vision_end|>", "")
                prompt = prompt.replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
                
                llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data, 'mm_processor_kwargs': video_kwargs}
                
                input_list.append(llm_inputs)
                save_dict.update({"messages": sample["messages"], "videos": video_path, "label": label, "label_res": label_res})
                metadata.append(save_dict)

            elif len(sample.get("images", [])) == 1:
                image_path = sample["images"][0]
                messages = [
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": sample["messages"][1]["content"]},
                    ]},
                ]
                
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                mm_data = {}
                if image_inputs is not None: mm_data["image"] = image_inputs
                if video_inputs is not None: mm_data["video"] = video_inputs
                
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt = prompt.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
                prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
                
                llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
                
                input_list.append(llm_inputs)
                save_dict.update({"messages": sample["messages"], "images": image_path, "label": label, "label_res": label_res})
                metadata.append(save_dict)

            else:
                messages = [
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": sample["messages"][1]["content"]},
                ]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
               
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs

                llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
                
                input_list.append(llm_inputs)
                save_dict.update({"messages": sample["messages"], "label": label, "label_res": label_res})
                metadata.append(save_dict)
        
        except Exception as e:
            print(f"Error processing input data index {i}: {e}")
            continue

   
    if len(input_list) > 0:
        outputs = vllm_model.generate(input_list, sampling_params=sampling_params)
        
        save_dict_list = []
        
        for id, item in enumerate(outputs):
            current_meta = metadata[id]
            error_count = 0
            
            for out in item.outputs:
                generation = out.text
                pred, pred_res = extract_labels(generation)

                if pred is None or pred_res is None:
                    error_count += 1
                    continue

                if pred != current_meta["label"] or pred_res != current_meta["label_res"]:
                    error_count += 1

            if error_count > 0:
                current_meta["error_count"] = error_count
                save_dict_list.append(current_meta)


        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for item in save_dict_list:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')


    del input_list
    del metadata
    del outputs
    gc.collect() 

print("Inference finished.")