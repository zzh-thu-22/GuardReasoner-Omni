<h2 align="center"><a href="https://arxiv.org/abs/2602.03328">GuardReasoner-Omni: A Reasoning-based Multi-modal Guardrail for Text, Image, Video, and Audio</a></h2>

<div align="center">

[ [📖 Paper](https://arxiv.org/abs/2602.03328) ] 
[ [🤗 GuardReasoner-Omni-3B](https://huggingface.co/zhu-thu-22/GuardReasoner-Omni-3B) ] 
[ [🤗 GuardReasoner-Omni-7B](https://huggingface.co/zhu-thu-22/GuardReasoner-Omni-7B) ] 
[ [🤗 data](https://huggingface.co/datasets/zhu-thu-22/GuardReasoner-Omni-data) ] 

</div>


## 🚀 Training

We first perform supervised fine-tuning on the GuardReasoner-OmniTrain-181k dataset to obtain the SFT model.

```bash
bash ./script/sft.sh
```

Next, run the following code to extract hard samples, which will be used to further refine the model.

```bash
python ./src/hard_sample_mining.py
```


Finally, conduct GRPO training on the mined hard samples using the previously obtained SFT model to produce the final GuardReasoner-Omni model.

```bash
bash ./script/GRPO.sh
```

For efficiency considerations, videos are downsampled to 1 FPS and capped at a maximum of 128 frames during training, with each frame processed at a maximum resolution of 64 × 28 × 28 pixels. Similarly, the resolution for the image modality is strictly constrained to a maximum of 2048 × 28 × 28 pixels.

## 🔮 Evaluation

Download the JSON files and mm_data, then put them in `./data/test_data` 

Run the following code to evaluate all benchmarks.

```bash
python ./src/generate.py
python ./src/evaluate.py
```


## Acknowledgement
Our method are partly based on the following resources. Thanks for their awesome works.
- [GuardReasoner](https://github.com/yueliu1999/GuardReasoner)
- [GuardReasoner-VL](https://github.com/yueliu1999/GuardReasoner-VL)
- [ms-swift](https://github.com/modelscope/ms-swift)
- [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni)



## Citations

If you find this repository helpful, please cite our paper.

```
@article{GuardReasoner,
  title={GuardReasoner: Towards Reasoning-based LLM Safeguards},
  author={Liu, Yue and Gao, Hongcheng and Zhai, Shengfang and Jun, Xia and Wu, Tianyi and Xue, Zhiwei and Chen, Yulin and Kawaguchi, Kenji and Zhang, Jiaheng and Hooi, Bryan},
  journal={arXiv preprint arXiv:2501.18492},
  year={2025}
}


@article{GuardReasoner-VL,
  title={GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning},
  author={Liu, Yue and Zhai, Shengfang and Du, Mingzhe and Chen, Yulin and Cao, Tri and Gao, Hongcheng and Wang, Cheng and Li, Xinfeng and Wang, Kun and Fang, Junfeng and Zhang, Jiaheng and Hooi, Bryan},
  journal={arXiv preprint arXiv:2505.11049},
  year={2025}
}

@misc{zhu2026guardreasoneromnireasoningbasedmultimodalguardrail,
      title={GuardReasoner-Omni: A Reasoning-based Multi-modal Guardrail for Text, Image, and Video}, 
      author={Zhenhao Zhu and Yue Liu and Yanpei Guo and Wenjie Qu and Cancan Chen and Yufei He and Yibo Li and Yulin Chen and Tianyi Wu and Huiying Xu and Xinzhong Zhu and Jiaheng Zhang},
      year={2026},
      eprint={2602.03328},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2602.03328}, 
}
```