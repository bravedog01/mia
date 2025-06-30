import os
import torch
import json
import math
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from sentence_transformers.util import dot_score
from eval import *
import numpy as np
import openai
from sentence_transformers import util
from openai import OpenAI
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig

client = OpenAI(
    base_url="https://yunwu.ai/v1",                  # 若使用 Yunwu.ai，可指定 base_url；否则默认 api.openai.com :contentReference[oaicite:1]{index=1}
    api_key="sk-IKK3x426pmLquyBLHTPlSN7mGBIALsUP1pxCwU8Ye5bwF2SG",             # 建议通过环境变量配置您的 Key :contentReference[oaicite:2]{index=2}
    timeout=200
)

os.environ['HF_ENDPOINT'] = 'hf-mirror.com'

def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data

def prepare_dataset(data, length):
    if data == "WikiMIA":
        ds = load_dataset(
                "parquet",
                data_files="./data/wikimia/WikiMIA_length32-00000-of-00001-*.parquet",
                split="train"
        )
        return ds

    elif data == "WikiMIA-paraphrased":
        original_dataset = load_dataset("zjysteven/WikiMIA_paraphrased_perturbed", split=f"WikiMIA_length{length}_paraphrased")
        original_dataset = convert_huggingface_data_to_list_dic(original_dataset)

    elif data == "WikiMIA-24":
        ds = load_dataset(
                "parquet",
                data_files="./data/WiKiMIA-24/WikiMIA_length32-00000-of-00001-*.parquet",
                split="train"
        )
        return ds

    else:
        original_non_member = []
        with open(f'data/MIMIR-ngram_7/non-members/{data}.jsonl', 'r') as f:
            for line in f:
                text = json.loads(line)
                original_non_member.append(" ".join(text.split()[:length]))
        
        original_member = []
        with open(f'data/MIMIR-ngram_7/members/{data}.jsonl', 'r') as f:
            for line in f:
                text = json.loads(line)
                original_member.append(" ".join(text.split()[:length]))
        original_member = original_member[:len(original_non_member)]

        testing_samples = 250
        size = min(testing_samples, len(original_non_member), len(original_member))
        original_dataset = [{"input":text, "label":1} for text in original_member[:size]] + [{"input":text, "label":0} for text in original_non_member[:size]]
        

    return original_dataset

def load_model(name1, name2):
    if "pythia-6.9b" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("/root/autodl-fs/pythia-6.9b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("/root/autodl-fs/pythia-6.9b")
    elif "pythia-2.8b" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
    elif "pythia-1.4b" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1.4b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    elif "pythia-160m" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("/root/autodl-fs/pythia-160m/onnx", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("/root/autodl-fs/pythia-160m")
    elif "pythia-6.9b-dedup" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")
    elif "pythia-2.8b-dedup" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b-deduped")
    elif "pythia-1.4b-dedup" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
    elif "pythia-160m-dedup" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")
    elif "llama2-13b" == name1:
        model1 = LlamaForCausalLM.from_pretrained("/root/autodl-fs/Llama-2-13b-chat-hf", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = LlamaTokenizer.from_pretrained("/root/autodl-fs/Llama-2-13b-chat-hf")
    elif "llama2-7b" == name1:
        model1 = LlamaForCausalLM.from_pretrained("/root/autodl-fs/Llama-2-7b-chat-hf", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = LlamaTokenizer.from_pretrained("/root/autodl-fs/Llama-2-7b-chat-hf")
    elif "falcon-7b" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/falcon-7b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("/root/autodl-fs/falcon-7b")
    elif "opt-6.7b" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/opt-6.7b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("/root/autodl-fs/opt-6.7b")
    elif "gpt2-xl" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/gpt2-xl", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("/root/autodl-fs/gpt2-xl")
    else:
        """
        You could modify the code here to make it compatible with other models
        """
        raise ValueError(f"Model {name1} is not currently supported!")

    if "pythia-6.9b" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("/root/autodl-fs/pythia-6.9b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("/root/autodl-fs/pythia-6.9b")
    elif name2 == "pythia-6.9b-rand":
        config = GPTNeoXConfig.from_pretrained("/root/autodl-fs/pythia-6.9b")
        model2 = GPTNeoXForCausalLM(config)      # 随机权重
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("/root/autodl-fs/pythia-6.9b")
    elif "pythia-2.8b" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
    elif "pythia-1.4b" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1.4b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    elif "pythia-160m" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("/root/autodl-fs/pythia-160m/onnx", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("/root/autodl-fs/pythia-160m") 
    elif name2 == "pythia-160m-rand":
        config = GPTNeoXConfig.from_pretrained("/root/autodl-fs/pythia-160m")
        model2 = GPTNeoXForCausalLM(config)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("/root/autodl-fs/pythia-160m")

    elif "pythia-6.9b-dedup" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")
    elif "pythia-2.8b-dedup" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b-deduped")
    elif "pythia-1.4b-dedup" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
    elif "pythia-160m-dedup" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped") 
    elif "llama2-13b" == name2:
        model2 = LlamaForCausalLM.from_pretrained("/root/autodl-fs/Llama-2-13b-chat-hf", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = LlamaTokenizer.from_pretrained("/root/autodl-fs/Llama-2-13b-chat-hf")
    elif "llama2-7b" == name2:
        model2 = LlamaForCausalLM.from_pretrained("/root/autodl-fs/Llama-2-7b-chat-hf", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = LlamaTokenizer.from_pretrained("/root/autodl-fs/Llama-2-7b-chat-hf")
    elif "falcon-7b" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/falcon-7b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("/root/autodl-fs/falcon-7b")
    elif "opt-6.7b" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/opt-6.7b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("/root/autodl-fs/opt-6.7b")
    elif "gpt2-xl" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("/root/autodl-fs/gpt2-xl", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("/root/autodl-fs/gpt2-xl")
    else:
        """
        You could modify the code here to make it compatible with other models
        """
        raise ValueError(f"Model {name2} is not currently supported!")

    return model1, model2, tokenizer1, tokenizer2


def calculatePerplexity(sentence, model, tokenizer, device):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()

def calculateTextSimilarity(model, embedding_model, tokenizer, text, decoding, device):

    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        
    input_ids = input_ids.to(device)
    
    sementic_similarity = []
    for i in range(1,input_ids.size(1)):
        input_ids_processed = input_ids[0][:i].unsqueeze(0)
        if decoding == "greedy":
            generation = model.generate(input_ids=input_ids_processed, attention_mask=torch.ones_like(input_ids_processed), do_sample=False, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        elif decoding == "nuclear":
            generation = model.generate(input_ids=input_ids_processed, attention_mask=torch.ones_like(input_ids_processed), do_sample=True, max_new_tokens=1, top_k=0, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        elif decoding == "contrastive":
            generation = model.generate(input_ids=input_ids_processed, attention_mask=torch.ones_like(input_ids_processed), penalty_alpha=0.6, max_new_tokens=1, top_k=4, pad_token_id=tokenizer.eos_token_id)
        
        generated_embedding = embedding_model.encode(tokenizer.decode(generation[0][-1]))
        label_embedding = embedding_model.encode([tokenizer.decode(input_ids[0][i])]) 
        score = dot_score(label_embedding, generated_embedding)[0].item()
        if score <= 0:
            score = 1e-16
        sementic_similarity.append(math.log(score))
    
    return  sementic_similarity


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def calculateTextSimilarity_api(
    embedding_model,
    tokenizer,
    text: str,
    decoding: str,
    device: str,
    *,
    api_model_name: str
):
    """
    通过 OpenAI API 预测每一步的下一个 token，并用本地 embedding_model 计算相似度。

    参数：
    - embedding_model: 本地嵌入模型（如 sentence-transformers）
    - tokenizer: 对应的 tokenizer
    - text: 待测文本
    - decoding: "greedy" | "nuclear" | "contrastive"
    - device: "cpu" 或 "cuda"
    - api_model_name: OpenAI Completion 模型名（必填）
    """
    # 1. 构造 input_ids
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    
    semantic_similarity = []
    for i in range(1, input_ids.size(1)):
        # 2. 上下文前缀
        prefix_ids = input_ids[0, :i].unsqueeze(0)
        decoded_prefix = tokenizer.decode(prefix_ids[0], skip_special_tokens=True)
        
        # 3. 调用 OpenAI Completion API 生成下一个 token
        params = {
            "model": api_model_name,
            "prompt":decoded_prefix,
            "max_tokens": 1,
            "temperature": 0.0 if decoding == "greedy" else 0.8
        }
        if decoding == "nuclear":
            params["top_p"] = 0.95
        if decoding == "contrastive":
            # API 端没有直接的 contrastive 解码，这里用采样 + top_p 近似
            params.update({"temperature": 1.0, "top_p": 0.9})
        
        resp=client.completions.create(
                  model="gpt-3.5-turbo-instruct",
                  prompt="The 2015 Nigerian Senate election in Bauchi State was held on March 28, 2015, to elect members of the Nigerian Senate to represent Bauchi State. Isah Hamma representing Bauchi Central, Malam Wakili",
                  max_tokens=1,
                  temperature=0
                )
        
        content = resp.choices[0].text
        
        # 4. 用本地 embedding_model 计算生成 token 与真实 token 的嵌入
        generated_embedding = embedding_model.encode(content)
        true_token = tokenizer.decode(input_ids[0, i], skip_special_tokens=True)
        label_embedding = embedding_model.encode(true_token)
        
        # 5. 计算余弦相似度并取对数
        score = dot_score(label_embedding, generated_embedding)[0].item()
        if score <= 0:
            score = 1e-16
        semantic_similarity.append(math.log(score))
    
    return semantic_similarity


def fitting(model, embedding_model, tokenizer, text, decoding):
    
    sementic_similarity = calculateTextSimilarity(model, embedding_model, tokenizer, text, decoding, device=model.device)
    p1, all_prob, p1_likelihood = calculatePerplexity(text, model, tokenizer, device=model.device)

    slope, intercept = np.polyfit(np.array(sementic_similarity), np.array(all_prob), 1)

    return slope, intercept