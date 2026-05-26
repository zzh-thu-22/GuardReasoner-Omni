import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import re
import json
import argparse
from template import INSTRUCTION
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_omni_utils import process_mm_info


def main():
    parser = argparse.ArgumentParser(description="GuardReasoner-Omni Inference")
    parser.add_argument("--model_path", type=str, default="./models/GuardReasoner-Omni-7B", help="model path")
    parser.add_argument("--benchmark_path", type=str, default="./data/test_data/", help="benchmark path")
    args = parser.parse_args()

    SAVE_PATH = f"./results/{args.model_path.split('/')[-1]}"

    vllm_model = LLM(model=args.model_path, gpu_memory_utilization=0.9, max_num_seqs=256, max_model_len=24576, tensor_parallel_size=1, limit_mm_per_prompt={"image": 1, "video": 1, "audio": 1})
    sampling_params = SamplingParams(temperature=0., top_p=1.0, max_tokens=4096)
    processor = AutoProcessor.from_pretrained(args.model_path)

    input_format = """{prompt}

AI assistant:
{response}
"""

    # text benchmark
    text_benchmark_list = ["Aegis_2_prompt", "Aegis_2_response", "BeaverTails", "HarmBench_prompt", 
    "HarmBench_response", "openaimoderation", "SimpleSafetyTest", "sorrybench", "toxic_chat_test",
    "wildguard_prompt", "wildguard_response", "xstest_prompt", "xstest_response", "saferlhf", "or-bench"]
    for idx, data_name in enumerate(text_benchmark_list):
        with open(f"{args.benchmark_path}{data_name}.json", 'r', encoding='utf-8') as file:
            data = json.load(file)

        file_path = f"{SAVE_PATH}/{data_name}/generated_predictions.jsonl"
        if os.path.exists(file_path):
            print(f"Results for {data_name} already exist. Skipping...")
            continue

        input_list = []
        label_list = []
        save_dict_list = []
        
        for i, sample in enumerate(data):
            save_dict = {}
            

            messages = [
                {
                    "role": "system",
                    "content": INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Human User:\n"
                        },
                        {
                            "type": "text", 
                            "text": input_format.format(prompt=sample["request"], response=sample["response"])
                        }
                    ],
                },
            ]

            mm_data = {}
        
            prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            # prompt = prompt.replace("\n<|im_end|>", "<|im_end|>\n<|im_start|>assistant")
            # prompt = prompt.replace("<|im_end|>\n<|im_start|>assistant\n<|im_start|>user", "\n<|im_end|>\n<|im_start|>user")
            # assert "<|im_end|>\n<|im_start|>assistant" in prompt
            # assert "\n<|im_end|>\n<|im_start|>user" in prompt
            
            llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
            input_list.append(llm_inputs)
            
            if sample["task_type"] == "request":
                label_list.append(sample["request_gt"])
            elif sample["task_type"] == "response":
                label_list.append(sample["response_gt"])
            else:
                raise ValueError("Invalid task type")

            
            save_dict["task_type"] = sample["task_type"]
            save_dict_list.append(save_dict)

            
        outputs = vllm_model.generate(input_list, sampling_params=sampling_params)
        
        for id, i in enumerate(outputs):
        
            try:
                if save_dict_list[id]["task_type"] == "request":
                    predict = re.search(r'Request:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
                elif save_dict_list[id]["task_type"] == "response":
                    predict = re.search(r'Response:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
                predict = "unharmful" if predict == "none" else predict
                predict = "unharmful" if predict == "harmless" else predict
            except:
                if label_list[id]=="unharmful":
                    predict = "harmful"
                else:
                    predict = "unharmful"

            save_dict_list[id]["text_output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            save_dict_list[id]["predict"] = predict
        
        
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in save_dict_list:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')
        
        



    ###image
    image_benchmark_list = ["LlavaGuard", "SPA-VL_prompt", "SPA-VL_response", "UnsafeBench", 
    "VLGuard", "holisafe"]
    for idx, data_name in enumerate(image_benchmark_list):
        with open(f"{args.benchmark_path}{data_name}.json", 'r', encoding='utf-8') as file:
            data = json.load(file)

        file_path = f"{SAVE_PATH}/{data_name}/generated_predictions.jsonl"
        if os.path.exists(file_path):
            print(f"Results for {data_name} already exist. Skipping...")
            continue

        input_list = []
        label_list = []
        
        save_dict_list = []
        
        for i, sample in enumerate(data):
            save_dict = {}
            
            messages = [
                {
                    "role": "system",
                    "content": INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Human User:\n"
                        },
                        {
                            "type": "image",
                            "image": sample['image_path'],
                            "max_pixels":2048*28*28
                        },
                        {
                            "type": "text", 
                            "text": input_format.format(prompt=sample["request"], response=sample["response"])
                        }
                    ],
                },
            ]

            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

            mm_data = {}
            assert images is not None, "Image inputs should not be None"
            mm_data["image"] = images
        
            prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            # prompt = prompt.replace("\n<|im_end|>", "<|im_end|>\n<|im_start|>assistant")
            # prompt = prompt.replace("<|im_end|>\n<|im_start|>assistant\n<|im_start|>user", "\n<|im_end|>\n<|im_start|>user")
            # assert "<|im_end|>\n<|im_start|>assistant" in prompt
            # assert "\n<|im_end|>\n<|im_start|>user" in prompt
            
            llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
            input_list.append(llm_inputs)
            
            if sample["task_type"] == "request":
                label_list.append(sample["request_gt"])
            elif sample["task_type"] == "response":
                label_list.append(sample["response_gt"])
            else:
                raise ValueError("Invalid task type")

            
            save_dict["task_type"] = sample["task_type"]
            save_dict_list.append(save_dict)

            
        outputs = vllm_model.generate(input_list, sampling_params=sampling_params)
        
        for id, i in enumerate(outputs):
        
            try:
                if save_dict_list[id]["task_type"] == "request":
                    predict = re.search(r'Request:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
                elif save_dict_list[id]["task_type"] == "response":
                    predict = re.search(r'Response:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
                predict = "unharmful" if predict == "none" else predict
                predict = "unharmful" if predict == "harmless" else predict
            except:
                if label_list[id]=="unharmful":
                    predict = "harmful"
                else:
                    predict = "unharmful"


            save_dict_list[id]["text_output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            save_dict_list[id]["predict"] = predict
        
        
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in save_dict_list:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')





    # # Video benchmark
    video_benchmark_list = ["FVC", "safesora", "ucf-crime_test", "XD-Violence_test", 'VA-SafetyBench_video']
    for idx, data_name in enumerate(video_benchmark_list):
        with open(f"{args.benchmark_path}{data_name}.json", 'r', encoding='utf-8') as file:
            data = json.load(file)

        file_path = f"{SAVE_PATH}/{data_name}/generated_predictions.jsonl"
        if os.path.exists(file_path):
            print(f"Results for {data_name} already exist. Skipping...")
            continue

        input_list = []
        label_list = []
        save_dict_list = []

        for i, sample in enumerate(data):
            save_dict = {}

            messages = [
                {
                    "role": "system",
                    "content": INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Human User:\n"
                        },
                        {
                            "type": "video", 
                            "video": sample['video_path'], 
                            "max_frames": 128, 
                            "min_frames": 1, 
                            "max_pixels":64*28*28,
                            "min_pixels":4*28*28,
                            "fps":1
                        },
                        {
                            "type": "text", 
                            "text": input_format.format(prompt=sample["request"], response=sample["response"])
                        }
                    ],
                },
            ]

            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

            mm_data = {}
            assert videos is not None, "Video inputs should not be None"
            mm_data["video"] = videos

            prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            # prompt = prompt.replace("\n<|im_end|>", "<|im_end|>\n<|im_start|>assistant")
            # prompt = prompt.replace("<|im_end|>\n<|im_start|>assistant\n<|im_start|>user", "\n<|im_end|>\n<|im_start|>user")
            # assert "<|im_end|>\n<|im_start|>assistant" in prompt
            # assert "\n<|im_end|>\n<|im_start|>user" in prompt


            llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
            input_list.append(llm_inputs)

            if sample["task_type"] == "request":
                label_list.append(sample["request_gt"])
            elif sample["task_type"] == "response":
                label_list.append(sample["response_gt"])
            else:
                raise ValueError("Invalid task type")
            
            save_dict["task_type"] = sample["task_type"]
            save_dict_list.append(save_dict)
            
        print(f"Total valid samples: {len(input_list)}")


        outputs = vllm_model.generate(input_list, sampling_params=sampling_params)

        for id, i in enumerate(outputs):
            
            try:
                if save_dict_list[id]["task_type"] == "request":
                    predict = re.search(r'Request:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
                elif save_dict_list[id]["task_type"] == "response":
                    predict = re.search(r'Response:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
                predict = "unharmful" if predict == "none" else predict
                predict = "unharmful" if predict == "harmless" else predict
            except:
                if label_list[id]=="unharmful":
                    predict = "harmful"
                else:
                    predict = "unharmful"

            save_dict_list[id]["text_output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            save_dict_list[id]["predict"] = predict


        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'w', encoding='utf-8') as f:
            for item in save_dict_list:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')



    ## audio benchmark
    audio_benchmark_list = ["AdvBench_prompt", "AdvBench_response", "Jailbreak-AudioBench",
    "Nemotron_prompt", "Nemotron_response", "mutox", 'VA-SafetyBench_audio']
    for idx, data_name in enumerate(audio_benchmark_list):
        with open(f"{args.benchmark_path}{data_name}.json", 'r', encoding='utf-8') as file:
            data = json.load(file)

        file_path = f"{SAVE_PATH}/{data_name}/generated_predictions.jsonl"
        if os.path.exists(file_path):
            print(f"Results for {data_name} already exist. Skipping...")
            continue

        input_list = []
        label_list = []
        save_dict_list = []

        for i, sample in enumerate(data):
            save_dict = {}

            messages = [
                {
                    "role": "system",
                    "content": INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Human User:\n"
                        },
                        {
                            "type": "audio", 
                            "audio": sample['audio_path']
                        },
                        {
                            "type": "text", 
                            "text": input_format.format(prompt=sample["request"], response=sample["response"])
                        }
                    ],
                },
            ]

            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
               
            mm_data = {}
            assert audios is not None, "Audio inputs should not be None"
            mm_data["audio"] = audios

            prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            # prompt = prompt.replace("\n<|im_end|>", "<|im_end|>\n<|im_start|>assistant")
            # prompt = prompt.replace("<|im_end|>\n<|im_start|>assistant\n<|im_start|>user", "\n<|im_end|>\n<|im_start|>user")
            # assert "<|im_end|>\n<|im_start|>assistant" in prompt
            # assert "\n<|im_end|>\n<|im_start|>user" in prompt


            llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
            input_list.append(llm_inputs)

            if sample["task_type"] == "request":
                label_list.append(sample["request_gt"])
            elif sample["task_type"] == "response":
                label_list.append(sample["response_gt"])
            else:
                raise ValueError("Invalid task type")
            
            save_dict["task_type"] = sample["task_type"]
            save_dict_list.append(save_dict)
            
        print(f"Total valid samples: {len(input_list)}")


        outputs = vllm_model.generate(input_list, sampling_params=sampling_params)

        for id, i in enumerate(outputs):
            
            try:
                if save_dict_list[id]["task_type"] == "request":
                    predict = re.search(r'Request:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
                elif save_dict_list[id]["task_type"] == "response":
                    predict = re.search(r'Response:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
                predict = "unharmful" if predict == "none" else predict
                predict = "unharmful" if predict == "harmless" else predict
            except:
                if label_list[id]=="unharmful":
                    predict = "harmful"
                else:
                    predict = "unharmful"

            save_dict_list[id]["text_output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            save_dict_list[id]["predict"] = predict


        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'w', encoding='utf-8') as f:
            for item in save_dict_list:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')


if __name__ == "__main__":
    main()