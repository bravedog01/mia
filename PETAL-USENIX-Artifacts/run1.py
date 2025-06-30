import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from options import Options
from vectors import Sentence_Transformer
from sentence_transformers import SentenceTransformer
from eval import *
from utils import *
import os
from openai import OpenAI
   
client = OpenAI(
    base_url="https://yunwu.ai/v1",                  # 若使用 Yunwu.ai，可指定 base_url；否则默认 api.openai.com :contentReference[oaicite:1]{index=1}
    api_key="sk-IKK3x426pmLquyBLHTPlSN7mGBIALsUP1pxCwU8Ye5bwF2SG",             # 建议通过环境变量配置您的 Key :contentReference[oaicite:2]{index=2}
    timeout=120
)
 
def inference(model1, model2, embedding_model, tokenizer1, tokenizer2,
              text, ex, decoding, is_llama):
    pred = {}

    # 1. 如果是 Llama2，用 API 单步生成并获取 logprob
    if is_llama:
        # 新：使用 Completions API 获取 logprobs
        resp = client.completions.create(
            model="llama-2-13b",  # 若不支持，可改为对应的 Instruct 模型
            prompt=text,                             # 使用 prompt 参数
            max_tokens=6,                            # 生成 1 个 token
            temperature=0.0,
            logprobs=True                               # 请求返回每个 token 的 log 概率
        )
         # 从 Completions 响应取出生成 token 与对数概率
        gen_token = resp.choices[0].text             # Completions API 下生成内容字段为 text :contentReference[oaicite:5]{index=5}
        logprob   = resp.choices[0].logprobs[0]      # Completions API 下 logprobs 数组直接位于 choices[].logprobs :contentReference[oaicite:6]{index=6}

        # 这里可以把 gen_token 和 logprob 暴露在 ex 中以便调试
        pred["PETAL"] = -logprob

    else:
        slope, intercept = fitting(
            model2, embedding_model, tokenizer2,
            text, decoding=decoding
        )
        text_similarity = calculateTextSimilarity(
            model1, embedding_model, tokenizer1,
            text, decoding=decoding,
            device=(None if is_llama else model1.device)
        )
        all_prob_estimated = [i*slope + intercept for i in text_similarity]
        pred["PETAL"] = -np.mean(all_prob_estimated).item()

    ex["pred"] = pred
    return ex

def evaluate_data(original_data, model1, model2, embedding_model, tokenizer1, tokenizer2, decoding,is_llama):
    print(f"all data size: {len(original_data)}")
    all_output = []
    for ex in tqdm(original_data): 
        text = ex["input"]
        new_ex = inference(model1, model2, embedding_model, tokenizer1, tokenizer2, text, ex, decoding,is_llama)
        all_output.append(new_ex)
    return all_output


if __name__ == '__main__':
    args = Options()
    args = args.parser.parse_args()
    os.environ['HF_ENDPOINT'] = 'hf-mirror.com'
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    args.output_dir = f"{args.output_dir}/{args.data}/{args.target_model}_{args.surrogate_model}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and data
    original_data = prepare_dataset(args.data, args.length)
    model1, model2, tokenizer1, tokenizer2 = load_model(args.target_model, args.surrogate_model)

    # load embedding model
    embedding_model = Sentence_Transformer(args.embedding_model,'cuda:0')


    all_output = evaluate_data(original_data, model1, model2, embedding_model, tokenizer1, tokenizer2, args.decoding,1)
    fig_fpr_tpr(all_output, args.output_dir)

