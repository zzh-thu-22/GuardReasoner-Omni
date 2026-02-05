# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any
import math

# Metadata
REWARD_NAME = "safety"
REWARD_TYPE = "batch"

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
    

def format_reward(response):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<result>.*?</result>"
    format_match = re.search(pattern, response, re.DOTALL)
    return 1.0 if format_match else 0.0


def accuracy_reward(res: str, ground_truth: str) -> float:

    request_pred, response_pred = extract_labels(res)
    request_gt, response_gt = extract_labels(ground_truth)

    request_reward = 0.0
    response_reward = 0.0

    if request_pred is None or request_gt is None:
        request_reward = 0.0
    elif request_pred == request_gt:
        request_reward = 0.5

    if response_pred is None or response_gt is None:
        response_reward = 0.0
    elif response_pred == response_gt:
        response_reward = 0.5
        
    return request_reward + response_reward


def exploration_reward(response_length: int, accuracy_score: float) -> float:
    """
    探索性长度奖励：
    1. 如果 accuracy_score == 1.0 (全对)，奖励为 0。
    2. 如果 accuracy_score < 1.0 (有错误)，鼓励写更长的 CoT。
    3. 使用 tanh 函数限制最大奖励，防止无限长度。
    """
    # 参数设置 (根据你的任务调整)
    MAX_REWARD = 0.2       # alpha: 只有当错了且写得够长时，最多能补回多少分
    TARGET_LENGTH = 300   # sigma: 期望的“足够长”的字符数 (大约 150-200 tokens)
    
    if accuracy_score >= 1.0:
        return 0.0
    
    # 计算长度得分 (0 到 1 之间)
    length_factor = math.tanh(response_length / TARGET_LENGTH)
    
    return MAX_REWARD * length_factor


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        response_length = reward_input["response_length"]
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        explore_score = exploration_reward(response_length, accuracy_score)

        overall = 0.0
        if format_score == 1.0:
            overall = accuracy_score + explore_score

        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "accuracy": accuracy_score,
                "exploration": explore_score
            }
        )

    return scores