import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_omni_utils import process_mm_info


def main():
    parser = argparse.ArgumentParser(description="GuardReasoner-Omni Hard Sample Mining")
    parser.add_argument("--model_path", type=str, default="./models/SFT_model", help="model path")
    args = parser.parse_args()

    MODEL_NAME = f"{args.model_path.split('/')[-1]}"
    OUTPUT_FILE = f"./hard_sample_{MODEL_NAME}.jsonl"


    vllm_model = LLM(model=args.model_path, gpu_memory_utilization=0.9, max_num_seqs=256, max_model_len=24576, tensor_parallel_size=1, limit_mm_per_prompt={"image": 1, "video": 1, "audio": 1})
    sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=4096, n=8)
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
    with open(f"./data/train_data/GuardReasoner_OmniTrain.jsonl", encoding="utf-8") as file:
        all_data = [json.loads(line) for line in file]

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
                        {"role": "system", "content": sample["messages"][0]["content"]},
                        {"role": "user", "content": [
                            {
                                "type": "video", 
                                "video": video_path, 
                                "max_frames": 128, 
                                "min_frames": 1, 
                                "max_pixels":64*28*28,
                                "min_pixels":4*28*28,
                                "fps":1
                            },
                            {"type": "text", "text": sample["messages"][1]["content"]},
                        ]},
                    ]
                    
                    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
                    
                    mm_data = {}
                    assert videos is not None
                    mm_data["video"] = videos
                    
                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    # prompt = prompt.replace("\n<|im_end|>", "<|im_end|>\n<|im_start|>assistant")
                    # prompt = prompt.replace("<|im_end|>\n<|im_start|>assistant\n<|im_start|>user", "\n<|im_end|>\n<|im_start|>user")
                    assert "<|vision_bos|><|VIDEO|><|vision_eos|>" in prompt
                    prompt = prompt.replace("<|vision_bos|><|VIDEO|><|vision_eos|>", "")
                    prompt = prompt.replace("<video>", "<|vision_bos|><|VIDEO|><|vision_eos|>")
                    
                    llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
                    
                    input_list.append(llm_inputs)
                    save_dict.update({"messages": sample["messages"], "video_path": video_path, "label": label, "label_res": label_res})
                    metadata.append(save_dict)

                elif len(sample.get("images", [])) == 1:
                    image_path = sample["images"][0]
                    messages = [
                        {"role": "system", "content": sample["messages"][0]["content"]},
                        {"role": "user", "content": [
                            {
                                "type": "image", 
                                "image": image_path,
                                "max_pixels":2048*28*28,
                            },
                            {"type": "text", "text": sample["messages"][1]["content"]},
                        ]},
                    ]
                    
                    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
                    mm_data = {}
        
                    assert images is not None
                    mm_data["image"] = images
                    
                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    # prompt = prompt.replace("\n<|im_end|>", "<|im_end|>\n<|im_start|>assistant")
                    # prompt = prompt.replace("<|im_end|>\n<|im_start|>assistant\n<|im_start|>user", "\n<|im_end|>\n<|im_start|>user")
                    assert "<|vision_bos|><|IMAGE|><|vision_eos|>" in prompt
                    prompt = prompt.replace("<|vision_bos|><|IMAGE|><|vision_eos|>", "")
                    prompt = prompt.replace("<image>", "<|vision_bos|><|IMAGE|><|vision_eos|>")
                    
                    llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
                    
                    input_list.append(llm_inputs)
                    save_dict.update({"messages": sample["messages"], "image_path": image_path, "label": label, "label_res": label_res})
                    metadata.append(save_dict)

                elif len(sample.get("audios", [])) == 1:
                    audio_path = sample["audios"][0]
                    messages = [
                        {"role": "system", "content": sample["messages"][0]["content"]},
                        {"role": "user", "content": [
                            {
                                "type": "audio", 
                                "audio": audio_path
                            },
                            {"type": "text", "text": sample["messages"][1]["content"]},
                        ]},
                    ]
                    
                    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
                    mm_data = {}
        
                    assert audios is not None
                    mm_data["audio"] = audios
                    
                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    # prompt = prompt.replace("\n<|im_end|>", "<|im_end|>\n<|im_start|>assistant")
                    # prompt = prompt.replace("<|im_end|>\n<|im_start|>assistant\n<|im_start|>user", "\n<|im_end|>\n<|im_start|>user")
                    assert "<|audio_bos|><|AUDIO|><|audio_eos|>" in prompt
                    prompt = prompt.replace("<|audio_bos|><|AUDIO|><|audio_eos|>", "")
                    prompt = prompt.replace("<audio>", "<|audio_bos|><|AUDIO|><|audio_eos|>")
                    
                    llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
                    
                    input_list.append(llm_inputs)
                    save_dict.update({"messages": sample["messages"], "audio_path": audio_path, "label": label, "label_res": label_res})
                    metadata.append(save_dict)
                
                else:
                    messages = [
                        {"role": "system", "content": sample["messages"][0]["content"]},
                        {"role": "user", "content": sample["messages"][1]["content"]},
                    ]
                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    # prompt = prompt.replace("\n<|im_end|>", "<|im_end|>\n<|im_start|>assistant")
                    # prompt = prompt.replace("<|im_end|>\n<|im_start|>assistant\n<|im_start|>user", "\n<|im_end|>\n<|im_start|>user")
                
                    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
                
                    mm_data = {}
                    assert images is None
                    assert videos is None
                    assert audios is None

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

                if error_count > 0 and error_count < len(item.outputs):
                    current_meta["error_count"] = error_count
                    current_meta["all_count"] = len(item.outputs)
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



if __name__ == "__main__":
    main()
