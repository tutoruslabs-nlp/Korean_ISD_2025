import argparse
from argparse import ArgumentTypeError
import json
import time

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data import (
    make_prompt,
    generation_config,
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about LLM Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--input", type=str, required=True, help="input filename")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--debug", type=str2bool, default=False)
# fmt: on

labels = {
    'inappropriate': 'inappropriate',
    'appropriate': 'appropriate',
    'default': 'unknown'
}

def main(args):
    # 모델 로딩
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f'----------------------------------------------------')
    print(f'model: {args.model_id}')
    print(f'input: {args.input}')
    print(f'----------------------------------------------------')
    
    # 입력 파일 로딩
    input_file = args.input
    with open(input_file, "r") as f:
        results = json.load(f)

    # 결과의 output을 clear
    for doc in results:
        doc["output"] = []

    text_count = 0
    stime = time.time()
    for idx, doc in enumerate(tqdm(results, desc="Inferencing")):
        docid = doc.get("id", "NIL")
        utterances= doc.get("input", {}).get("utterance", [])
        for data in utterances:
            uid = data.get("id", "")
            text = data.get("form", "")
            
            # 프롬프트 생성
            prompt = make_prompt(text, tokenizer, flag_vlm=False)
            
            # 질문에 대한 응답 생성
            model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            terminators = list(filter(lambda x: x is not None, [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]))
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=128,
                eos_token_id=terminators,
                generation_config=generation_config,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["<|endofturn|>", "<|stop|>"],
                tokenizer=tokenizer,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 응답 확인
            response = response.strip()
            response_str = response.replace("\n", "\\n")
            if False and args.debug: print(f'[DBG] response: [{response_str}]')

            if response not in labels:
                response = labels.get('default')
                if args.debug: print(f'[기본값 적용] {text} -> {response}]')

            # 예측결과 저장
            tmp = {"id": uid, "label": response}
            results[idx]["output"].append(tmp)
            
            # 디버깅용 출력
            if False and args.debug: print(f'{text} -> {response}')
            text_count += 1

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))

    response_time = time.time() - stime
    avg_time = round(float(response_time/text_count), 3)
    print(f'텍스트 수량: {text_count}')
    print(f'예측시간(전체): {response_time}')
    print(f'예측시간(평균): {avg_time}')


if __name__ == "__main__":
    exit(main(parser.parse_args()))
