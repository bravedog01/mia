import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from options import Options
from vectors import Sentence_Transformer
from eval import *
from utils import *
     
def inference(model1, model2, embedding_model, tokenizer1, tokenizer2, text, ex, decoding):
    pred = {}
    slope, intercept = fitting(model2, embedding_model, tokenizer2, text, decoding=decoding)
    text_similarity = calculateTextSimilarity(model1, embedding_model, tokenizer1, text, decoding=decoding, device=model1.device)
    all_prob_estimated = [i*slope + intercept for i in text_similarity]
    pred["PETAL"] = -np.mean(all_prob_estimated).item()
    ex["pred"] = pred
    return ex

def inference_api(model2, embedding_model, tokenizer1, tokenizer2, text, ex, decoding):
    pred = {}
    slope, intercept = fitting(model2, embedding_model, tokenizer2, text, decoding=decoding)
    text_similarity = calculateTextSimilarity_api(embedding_model, tokenizer1, text, decoding=decoding, device="cuda:0",api_model_name="gpt-3.5-turbo-instruct")
    all_prob_estimated = [i*slope + intercept for i in text_similarity]
    pred["PETAL"] = -np.mean(all_prob_estimated).item()
    ex["pred"] = pred
    return ex


def evaluate_data(original_data, model1, model2, embedding_model, tokenizer1, tokenizer2, decoding):
    print(f"all data size: {len(original_data)}")
    all_output = []
    for ex in tqdm(original_data): 
        text = ex["input"]
        new_ex = inference(model1, model2, embedding_model, tokenizer1, tokenizer2, text, ex, decoding)
        all_output.append(new_ex)
    return all_output

def evaluate_data_api(original_data, model2, embedding_model, tokenizer1, tokenizer2, decoding):
    print(f"all data size: {len(original_data)}")
    all_output = []
    for ex in tqdm(original_data): 
        text = ex["input"]
        new_ex = inference_api(model2, embedding_model, tokenizer1, tokenizer2, text, ex, decoding)
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
    embedding_model = Sentence_Transformer(args.embedding_model, model1.device)
    #inputs = original_data['input'][77:79]+original_data['input'][0:49]
    #labels = original_data['label'][77:79]+original_data['label'][0:49]
    inputs = original_data['input'][:100]
    labels = original_data['label'][:100]

    # 构造 list of dicts
    examples = [
    {'input': txt, 'label': lbl}
    for txt, lbl in zip(inputs, labels)
    ]

    all_output = evaluate_data_api(examples, model2, embedding_model, tokenizer1, tokenizer2, args.decoding)
    fig_fpr_tpr(all_output, args.output_dir)

