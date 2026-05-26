import pandas as pd
import os
from sklearn.metrics import f1_score
import openpyxl

folders = [
"./results/GuardReasoner-Omni-3B",
"./results/GuardReasoner-Omni-7B"
]


def cal_f1(folder="./results/7B/", dataset="ToxicChat", target_list=None):

    file_name = folder + "/" + dataset + "/generated_predictions.jsonl"
    if not os.path.exists(file_name):
        return 0

    pred = pd.read_json(file_name, lines=True)

    labels = []
    pred_labels = []

    for i in range(len(pred)):
        gt_example = pred['label'][i]
        pred_example = pred['predict'][i]
        assert gt_example in ["harmful", "unharmful"]
        assert pred_example in ["harmful", "unharmful"]
        labels.append(gt_example)
        pred_labels.append(pred_example)
        
    if len(labels) == 0:
        return 0
    
    prompt_f1 = f1_score(labels, pred_labels, pos_label='harmful')
    prompt_f1 *= 100

 
    if target_list is not None:
        target_list.append({"f1": prompt_f1, "weight": len(labels)})
        
    return prompt_f1


def cal_weighted_f1(f1_list):
    if not f1_list:
        return 0
    
    total_weight = sum(item["weight"] for item in f1_list)
    if total_weight == 0:
        return 0
        
    weighted_f1 = sum(item["f1"] * item["weight"] for item in f1_list) / total_weight
    
    return weighted_f1


def adjust_excel_style(df, file_path):
    sheet_name = 'Sheet1'
    combined_df = df

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        combined_df.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for col in worksheet.columns:
            max_length = 0
            column_letter = col[0].column_letter 
            for cell in col:
                if cell.value:
                    val_len = len(str(cell.value))
                    if val_len > max_length:
                        max_length = val_len
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column_letter].width = adjusted_width

        
all_model_prompt_results = []
all_model_response_results = []

for folder in folders:
    
    prompt_text_list = []
    prompt_image_list = []
    prompt_video_list = []
    prompt_audio_list = []

    # 1. Text
    toxic_chat_f1 = cal_f1(folder, "toxic_chat_test", prompt_text_list)
    harm_bench_prompt_f1 = cal_f1(folder, "HarmBench_prompt", prompt_text_list)
    openai_moderation_f1 = cal_f1(folder, "openaimoderation", prompt_text_list)
    aegis_2_safety_test_f1 = cal_f1(folder, "Aegis_2_prompt", prompt_text_list)
    simple_safety_tests_f1 = cal_f1(folder, "SimpleSafetyTest", prompt_text_list)
    sorrybench_f1 = cal_f1(folder, "sorrybench", prompt_text_list)
    wild_guard_test_prompt_f1 = cal_f1(folder, "wildguard_prompt", prompt_text_list)
    xstest_prompt_harmful_f1 = cal_f1(folder, "xstest_prompt", prompt_text_list)
    or_bench_f1 = cal_f1(folder, "or-bench", prompt_text_list)
    
    weighted_f1_prompt_text = cal_weighted_f1(prompt_text_list)

    # 2. Image 
    LlavaGuard_f1 = cal_f1(folder, "LlavaGuard", prompt_image_list)
    unsafe_bench_f1 = cal_f1(folder, "UnsafeBench", prompt_image_list)
    VLGuard_f1 = cal_f1(folder, "VLGuard", prompt_image_list)
    spa_vl_f1_prompt = cal_f1(folder, "SPA-VL_prompt", prompt_image_list)
    holisafe_f1 = cal_f1(folder, "holisafe", prompt_image_list)
    
    weighted_f1_prompt_image = cal_weighted_f1(prompt_image_list)

    # 3. Video 
    ucf_crime_test_f1 = cal_f1(folder, "ucf-crime_test", prompt_video_list)
    XD_Violence_test_f1 = cal_f1(folder, "XD-Violence_test", prompt_video_list)
    FVC_f1 = cal_f1(folder, "FVC", prompt_video_list)
    safesora_f1 = cal_f1(folder, "safesora", prompt_video_list)
    VA_SafetyBench_video_f1 = cal_f1(folder, "VA-SafetyBench_video", prompt_video_list)

    weighted_f1_prompt_video = cal_weighted_f1(prompt_video_list)

    # 4. Audio 
    AdvBench_prompt_f1 = cal_f1(folder, "AdvBench_prompt", prompt_audio_list)
    Jailbreak_AudioBench_f1 = cal_f1(folder, "Jailbreak-AudioBench", prompt_audio_list)
    Nemotron_prompt_f1 = cal_f1(folder, "Nemotron_prompt", prompt_audio_list)
    mutox_prompt_f1 = cal_f1(folder, "mutox", prompt_audio_list)
    VA_SafetyBench_audio_f1 = cal_f1(folder, "VA-SafetyBench_audio", prompt_audio_list)
    
    weighted_f1_prompt_audio = cal_weighted_f1(prompt_audio_list)

    # 5. average
    all_prompt_list = prompt_text_list + prompt_image_list + prompt_video_list + prompt_audio_list
    weighted_f1_prompt_all = cal_weighted_f1(all_prompt_list)

    model_prompt_entry = {
        "model": folder.split("/")[-1],
        "toxic_chat": toxic_chat_f1,
        "HarmBench": harm_bench_prompt_f1,
        "openai_moderation": openai_moderation_f1,
        "aegis_2.0_safety": aegis_2_safety_test_f1,
        "simple_safety_tests": simple_safety_tests_f1,
        "wildguard": wild_guard_test_prompt_f1,
        "sorrybench": sorrybench_f1,
        "xstest": xstest_prompt_harmful_f1,
        "or-bench": or_bench_f1,
        "weighted_text": weighted_f1_prompt_text,     
        
        "LlavaGuard": LlavaGuard_f1,
        "unsafe_bench": unsafe_bench_f1,
        "VLGuard": VLGuard_f1,
        "spa_vl": spa_vl_f1_prompt,
        "holisafe": holisafe_f1,
        "weighted_image": weighted_f1_prompt_image,   
        
        "ucf_crime": ucf_crime_test_f1,
        "XD_Violence": XD_Violence_test_f1,
        "FVC": FVC_f1,
        "safesora": safesora_f1,
        "VA-SafetyBench_video": VA_SafetyBench_video_f1,
        "weighted_video": weighted_f1_prompt_video,  
        
        "AdvBench": AdvBench_prompt_f1,
        "Jailbreak-AudioBench": Jailbreak_AudioBench_f1,
        "Nemotron": Nemotron_prompt_f1,
        "mutox": mutox_prompt_f1,
        "VA-SafetyBench_audio": VA_SafetyBench_audio_f1,
        "weighted_audio": weighted_f1_prompt_audio, 
        
        "weighted_prompt_total": weighted_f1_prompt_all 
    }
    all_model_prompt_results.append(model_prompt_entry)
    

    # ================= Response =================
    response_text_list = []
    response_image_list = []
    response_audio_list = []

    aegis_2_response_f1 = cal_f1(folder, "Aegis_2_response", response_text_list)
    harm_bench_response_f1 = cal_f1(folder, "HarmBench_response", response_text_list)
    safe_rlhf_response_f1 = cal_f1(folder, "saferlhf", response_text_list)
    beaver_tails_response_f1 = cal_f1(folder, "BeaverTails", response_text_list)
    wild_guard_test_reponse_f1 = cal_f1(folder, "wildguard_response", response_text_list)
    
    weighted_f1_response_text = cal_weighted_f1(response_text_list)

    spa_vl_f1_response = cal_f1(folder, "SPA-VL_response", response_image_list)
    weighted_f1_response_image = cal_weighted_f1(response_image_list)

    AdvBench_response_f1 = cal_f1(folder, "AdvBench_response", response_audio_list)
    Nemotron_response_f1 = cal_f1(folder, "Nemotron_response", response_audio_list)
    weighted_f1_response_audio = cal_weighted_f1(response_audio_list)

    all_response_list = response_text_list + response_image_list + response_audio_list
    weighted_f1_response_all = cal_weighted_f1(all_response_list)

    model_response_entry = {
        "model": folder.split("/")[-1],
        "HarmBench": harm_bench_response_f1,
        "aegis_2.0": aegis_2_response_f1,
        "saferlhf": safe_rlhf_response_f1,
        "wildguard": wild_guard_test_reponse_f1,
        "beaver_tails": beaver_tails_response_f1,
        "weighted_text": weighted_f1_response_text,
        
        "spa-vl": spa_vl_f1_response,
        "weighted_image": weighted_f1_response_image,

        "AdvBench": AdvBench_response_f1,
        "Nemotron": Nemotron_response_f1,
        "weighted_audio": weighted_f1_response_audio,
        
        "weighted_response_total": weighted_f1_response_all
    }
    all_model_response_results.append(model_response_entry)
   


prompt_df = pd.DataFrame(all_model_prompt_results)
response_df = pd.DataFrame(all_model_response_results)

adjust_excel_style(prompt_df, "prompt_results.xlsx")
adjust_excel_style(response_df, "response_results.xlsx")