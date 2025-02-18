We used [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) library to perform supervised fine-tuning of the LLM. There are three different experimental configurations, each using a varying number of training samples to evaluate the impact of dataset size on fine-tuning performance.


1. Setup `LLaMA-Factory` by following the instructions [here](https://github.com/hiyouga/LLaMA-Factory).

2. Fine-tune the model with the following command:

```sh
llamafactory-cli train fine_tuning/train_lora/erx-llama-3-8b-low.yaml
```

3. Chat with the model:

```sh
llamafactory-cli chat fine_tuning/inference/erx-llama-3-8b-low.yaml
```

4. Run inference server:

```sh
export CUDA_VISIBLE_DEVICES=3
export API_PORT=8008
llamafactory-cli api fine_tuning/examples/inference/erx-llama-3-8b-low.yaml
```

5. Publish the model to Hugging Face Hub:

```sh
huggingface-cli upload erx-llama-3-8b-low fine_tuning/saves/erx-llama-3-8b-low
```