import os
import re
import json
import argparse
from template import INSTRUCTION
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser(description="GuardReasoner-Omni Inference")
parser.add_argument("--model_path", type=str, default="GuardReasoner-Omni-4B", help="model path")
parser.add_argument("--benchmark_path", type=str, default="./data/test/", help="benchmark path")
args = parser.parse_args()


SAVE_PATH = f"./eval/results/{args.model_path.split('/')[-1]}"

vllm_model = LLM(model=args.model_path, gpu_memory_utilization=0.9, max_num_seqs=256, max_model_len=32768, limit_mm_per_prompt={"image": 10, "video": 10})
sampling_params = SamplingParams(temperature=0., top_p=1.0, max_tokens=4096)
processor = AutoProcessor.from_pretrained(args.model_path)


# text benchmark
# prompt harmfulness detection 6 datasets
for idx, data_name in enumerate(["ToxicChat", "HarmBenchPrompt", "OpenAIModeration", "AegisSafetyTest", "SimpleSafetyTests", "WildGuardTestPrompt"]):
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
                    {"type": "text", "text": sample['input']},
                ],
            },
        ]

        mm_data = {}
    
        prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        
        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
        input_list.append(llm_inputs)
        

        label_list.append(sample['output'])
        save_dict["text_input"] = sample['input']
        save_dict_list.append(save_dict)


    if data_name == "WildGuardTestPrompt": 
        label_list = [re.search(r'Request:\s*(harmful|unharmful|none|harmless)', item, re.IGNORECASE).groups()[0].lower() for item in label_list]
        label_list = ["unharmful" if item == "none" else item for item in label_list]
        label_list = ["unharmful" if item == "harmless" else item for item in label_list]
        
    outputs = vllm_model.generate(input_list, sampling_params=sampling_params)

    predict_list = []
    len_list = []
    for id, i in enumerate(outputs):
        response_len = len(i.outputs[0].text)
        
        try:
            predict = re.search(r'Request:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
            predict = "unharmful" if predict == "none" else predict
            predict = "unharmful" if predict == "harmless" else predict
        except:
            save_dict_list[id]["text_output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            if label_list[id]=="unharmful":
                predict = "harmful"
            else:
                predict = "unharmful"
            predict_list.append(predict)
            save_dict_list[id]["predict"] = predict
            
            continue
        predict_list.append(predict)
        len_list.append(response_len)
        save_dict_list[id]["text_output"] = i.outputs[0].text
        save_dict_list[id]["label"] = label_list[id]
        save_dict_list[id]["predict"] = predict
        save_dict_list[id]["res_len"] = response_len
    
    
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in save_dict_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    


# text benchmark
# response harmfulness detection 5 datasets
for idx, data_name in enumerate(["HarmBenchResponse", "SafeRLHF", "BeaverTails", "XSTestReponseHarmful", "WildGuardTestResponse"]):
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
                    {"type": "text", "text": sample['input']},
                ],
            },
        ]
                                    
        mm_data = {}
    
        prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
        input_list.append(llm_inputs)
        
        label_list.append(sample['output'])
        save_dict["text_input"] = sample['input']
        save_dict_list.append(save_dict)


    outputs = vllm_model.generate(input_list, sampling_params=sampling_params)
    

    predict_list = []
    len_list = []
    if data_name in ["WildGuardTestResponse", "SafeRLHF", "BeaverTails"]: 
        label_list = [re.search(r'Response:\s*(harmful|unharmful|none|harmless)', item, re.IGNORECASE).groups()[0].lower() for item in label_list]
        label_list = ["unharmful" if item == "none" else item for item in label_list]
        label_list = ["unharmful" if item == "harmless" else item for item in label_list]

        
    for id, i in enumerate(outputs):
        response_len = len(i.outputs[0].text)
        
        try:
            predict = re.search(r'Response:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
            predict = "unharmful" if predict == "none" else predict
            predict = "unharmful" if predict == "harmless" else predict
            predict_list.append(predict)
        except:
            save_dict_list[id]["text_output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            if label_list[id]=="unharmful":
                predict = "harmful"
            else:
                predict = "unharmful"
            predict_list.append(predict)
            save_dict_list[id]["predict"] = predict
            continue
        
        len_list.append(response_len)
        
        save_dict_list[id]["text_output"] = i.outputs[0].text
        save_dict_list[id]["label"] = label_list[id]
        save_dict_list[id]["res_len"] = response_len
        

    for id, i in enumerate(predict_list):
        save_dict_list[id]["predict"] = i

    
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in save_dict_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')



GuardReasoner_VLTest = load_dataset("yueliu1999/GuardReasoner-VLTest")
HarmfulImageTest = GuardReasoner_VLTest.filter(lambda example: example['label'] == 0)
SPA_VL_Eval = GuardReasoner_VLTest.filter(lambda example: example['label'] == 1)

# for image benchmark
# prompt harmfulness detection 
for idx, data_name in enumerate(["HarmImageTest"]):
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
    
        try:
            label = re.search(r'Request:\s*(harmful|unharmful|none|harmless)', sample['messages'][1]['content'], re.IGNORECASE).groups()[0].lower()
            label = "unharmful" if label == "none" else label
            label = "unharmful" if label == "harmless" else label
            label_list.append(label)
        except:
            pass


        input = sample['messages'][0]['content']
        input = input[input.find("\n\nHuman user:")+2:]
        
        messages = [
            {
                "role": "system",
                "content": INSTRUCTION,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": HarmfulImageTest['test'][i]['image'],
                    },
                    {"type": "text", "text": input},
                ],
            },
        ]

            
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True
        )
        
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
    
        prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

        prompt = prompt.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
        prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
        
        input_list.append(llm_inputs)
        
        save_dict["text_input"] = INSTRUCTION + input
        
        save_dict_list.append(save_dict)
    

    
    outputs = vllm_model.generate(input_list, sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    
    len_list = []
    for id, i in enumerate(outputs):
        response_len = len(i.outputs[0].text)
        
        try:
            predict = re.search(r'Request:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
            predict = "unharmful" if predict == "none" else predict
            predict = "unharmful" if predict == "harmless" else predict
        except:
            save_dict_list[id]["text_output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            if label_list[id]=="unharmful":
                predict = "harmful"
            else:
                predict = "unharmful"
            save_dict_list[id]["predict"] = predict
            continue
        
        len_list.append(response_len)
        save_dict_list[id]["text_output"] = i.outputs[0].text
        save_dict_list[id]["label"] = label_list[id]
        save_dict_list[id]["predict"] = predict
        save_dict_list[id]["res_len"] = response_len
    
    
    file_path = f"{SAVE_PATH}/{data_name}/generated_predictions.jsonl"
    
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in save_dict_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    

    



# # for text-image benchmark
# # prompt harmfulness detection
# # response harmfulness detection
for idx, data_name in enumerate(["SPA_VL_Eval"]):
    with open(f"{args.benchmark_path}{data_name}.json", 'r', encoding='utf-8') as file:
        data = json.load(file)

    file_path = f"{SAVE_PATH}/{data_name}/generated_predictions.jsonl"
    if os.path.exists(file_path):
        print(f"Results for {data_name} already exist. Skipping...")
        continue
        

    input_list = []
    label_list = []
    label_list_res = []
    save_dict_list = []
    

    for i, sample in enumerate(data):
        save_dict = {}
        
        input = sample['messages'][0]['content']
        input = input[input.find("\n\nHuman user:")+2:]
        
    
        try:
            label = re.search(r'Request:\s*(harmful|unharmful|none|harmless)', sample['messages'][1]['content'], re.IGNORECASE).groups()[0].lower()
            label = "unharmful" if label == "none" else label
            label = "unharmful" if label == "harmless" else label
            label_list.append(label)

            label_res = re.search(r'Response:\s*(harmful|unharmful|none|harmless)', sample['messages'][1]['content'], re.IGNORECASE).groups()[0].lower()
            label_res = "unharmful" if label_res == "none" else label_res
            label_res = "unharmful" if label_res == "harmless" else label_res
            label_list_res.append(label_res)
        except:
            pass
        
        messages = [
            {
                "role": "system",
                "content": INSTRUCTION,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": SPA_VL_Eval['test'][i]['image'],
                    },
                    {"type": "text", "text": input},
                ],
            },
        ]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
    
        prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

        prompt = prompt.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
        prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
        
        input_list.append(llm_inputs)
        
        save_dict["text_input"] = INSTRUCTION + input
        
        save_dict_list.append(save_dict)
    
    
    outputs = vllm_model.generate(input_list, sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    len_list_prompt = []
    len_list_res = []
    for id, i in enumerate(outputs):
        response_len = len(i.outputs[0].text)
        
        try:
            predict = re.search(r'Request:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
            predict = "unharmful" if predict == "none" else predict
            predict = "unharmful" if predict == "harmless" else predict
        except:
            save_dict_list[id]["text_output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            if label_list[id]=="unharmful":
                predict = "harmful"
            else:
                predict = "unharmful"
            save_dict_list[id]["predict"] = predict
            continue
        
        len_list_prompt.append(response_len)
        
        save_dict_list[id]["text_output"] = i.outputs[0].text
        save_dict_list[id]["label"] = label_list[id]
        save_dict_list[id]["predict"] = predict
        save_dict_list[id]["res_len"] = response_len
    

    for id, i in enumerate(outputs):
        response_len = len(i.outputs[0].text)
        
        try:
            predict_res = re.search(r'Response:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
            predict_res = "unharmful" if predict_res == "none" else predict_res
            predict_res = "unharmful" if predict_res == "harmless" else predict_res
        except:
            save_dict_list[id]["text_output"] = i.outputs[0].text
            save_dict_list[id]["label_res"] = label_list_res[id]
            if label_list_res[id]=="unharmful":
                predict_res = "harmful"
            else:
                predict_res = "unharmful"
            save_dict_list[id]["predict_res"] = predict_res
            continue

        len_list_res.append(response_len)
        
        save_dict_list[id]["text_output"] = i.outputs[0].text
        save_dict_list[id]["label_res"] = label_list_res[id]
        save_dict_list[id]["predict_res"] = predict_res
        save_dict_list[id]["res_len"] = response_len
        
    
    
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in save_dict_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    





input_format = """{prompt}

AI assistant:
{response}
"""

# Video benchmark
# prompt harmfulness detection
for idx, data_name in enumerate(["ucf-crime_test", "XD-Violence_test", "HarmVideo", "FVC"]):
    with open(f"{args.benchmark_path}{data_name}.json", 'r', encoding='utf-8') as file:
        data = json.load(file)

    file_path = f"{SAVE_PATH}/{data_name}/generated_predictions.jsonl"
    if os.path.exists(file_path):
        print(f"Results for {data_name} already exist. Skipping...")
        continue

    input_list = []
    label_list = []
    save_dict_list = []

    count = 0

    for i, sample in enumerate(data):
        save_dict = {}
        video_path = sample['videos']

        try:
            label_list.append(sample["request_gt"].lower())
        except:
            pass
        
        messages = [
            {
                "role": "system",
                "content": INSTRUCTION,
            },
            {
                "role": "user",
                "content": [
                {"type": "text", "text": "Human user:\n"},
                {"type": "video", "video": video_path, "max_frames": 128, "max_pixels":128*32*32},
                {"type": "text", "text": input_format.format(prompt=sample["request"], response="None")}
                ],
            },
        ]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        if video_inputs is not None:
            mm_data["video"] = video_inputs

        prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data, 'mm_processor_kwargs': video_kwargs}
        
        input_list.append(llm_inputs)
        
        save_dict["text_input"] = INSTRUCTION
        
        save_dict_list.append(save_dict)
        
    print(f"Total valid samples: {len(input_list)}")


    outputs = vllm_model.generate(input_list, sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text


    len_list_prompt = []

    for id, i in enumerate(outputs):
        response_len = len(i.outputs[0].text)
        
        try:
            predict = re.search(r'Request:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
            predict = "unharmful" if predict == "none" else predict
            predict = "unharmful" if predict == "harmless" else predict
        except:
            save_dict_list[id]["text_output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            if label_list[id]=="unharmful":
                predict = "harmful"
            else:
                predict = "unharmful"
            save_dict_list[id]["predict"] = predict
            continue
        
        len_list_prompt.append(response_len)
        
        save_dict_list[id]["text_output"] = i.outputs[0].text
        save_dict_list[id]["label"] = label_list[id]
        save_dict_list[id]["predict"] = predict
        save_dict_list[id]["res_len"] = response_len


   

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in save_dict_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')









# for text-video benchmark
# prompt harmfulness detection
# response harmfulness detection
for idx, data_name in enumerate(["HarmTextVideo"]):
    with open(f"{args.benchmark_path}{data_name}.json", 'r', encoding='utf-8') as file:
        data = json.load(file)

        
    file_path = f"{SAVE_PATH}/{data_name}/generated_predictions.jsonl"
    if os.path.exists(file_path):
        print(f"Results for {data_name} already exist. Skipping...")
        continue


    input_list = []
    label_list = []
    label_list_res = []
    save_dict_list = []
    

    for i, sample in enumerate(data):
        save_dict = {}
        video_path = sample['videos']    
    
        try:
            label_list.append(sample["request_gt"].lower())
            label_list_res.append(sample["response_gt"].lower())
        except:
            pass
        
        messages = [
            {
                "role": "system",
                "content": INSTRUCTION,
            },
            {
                "role": "user",
                "content": [
                {"type": "text", "text": "Human user:\n"},
                {"type": "video", "video": video_path, "max_frames": 128, "max_pixels":128*32*32},
                {"type": "text", "text": input_format.format(prompt=sample["request"], response=sample["response"])}
                ],
            },
        ]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        if video_inputs is not None:
            mm_data["video"] = video_inputs
    
        prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data, 'mm_processor_kwargs': video_kwargs}
        
        input_list.append(llm_inputs)
        
        save_dict["text_input"] = INSTRUCTION
        
        save_dict_list.append(save_dict)
    
    
    outputs = vllm_model.generate(input_list, sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    predict_list = []
    predict_list_res = []
    
    len_list_prompt = []
    len_list_res = []
    for id, i in enumerate(outputs):
        response_len = len(i.outputs[0].text)
        
        try:
            predict = re.search(r'Request:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
            predict = "unharmful" if predict == "none" else predict
            predict = "unharmful" if predict == "harmless" else predict
        except:
            save_dict_list[id]["text_output"] = i.outputs[0].text
            save_dict_list[id]["label"] = label_list[id]
            if label_list[id]=="unharmful":
                predict = "harmful"
            else:
                predict = "unharmful"
            save_dict_list[id]["predict"] = predict
            continue
        
        
        len_list_prompt.append(response_len)
        
        save_dict_list[id]["text_output"] = i.outputs[0].text
        save_dict_list[id]["label"] = label_list[id]
        save_dict_list[id]["predict"] = predict
        save_dict_list[id]["res_len"] = response_len
    

    for id, i in enumerate(outputs):
        response_len = len(i.outputs[0].text)
        
        try:
            predict_res = re.search(r'Response:\s*(harmful|unharmful|none|harmless)', i.outputs[0].text, re.IGNORECASE).groups()[0].lower()
            predict_res = "unharmful" if predict_res == "none" else predict_res
            predict_res = "unharmful" if predict_res == "harmless" else predict_res
        except:
            save_dict_list[id]["text_output"] = i.outputs[0].text
            save_dict_list[id]["label_res"] = label_list_res[id]
            if label_list_res[id]=="unharmful":
                predict_res = "harmful"
            else:
                predict_res = "unharmful"
            save_dict_list[id]["predict_res"] = predict_res
            continue
        
        len_list_res.append(response_len)
        
        save_dict_list[id]["text_output"] = i.outputs[0].text
        save_dict_list[id]["label_res"] = label_list_res[id]
        save_dict_list[id]["predict_res"] = predict_res
        save_dict_list[id]["res_len"] = response_len
        
    
    
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in save_dict_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')