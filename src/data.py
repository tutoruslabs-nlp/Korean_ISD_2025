from transformers import GenerationConfig


generation_config = GenerationConfig(
    do_sample=False,
    # temperature=0.1,
    # top_p=0.95,
    seed=42,
)


def make_prompt(text, tokenizer, flag_vlm):
    instruction = (
        "[지침]\n[입력 문장]에 대해서 적합한 label을 생성하시오. [생성 기준]을 꼼꼼히 읽고 이해하는 것이 중요합니다.\n\n"
        "[생성 기준]\n"
        " - 당신은 문장에 대한 label을 분류하는 챗봇입니다.\n"
        " - label의 종류는 'appropriate', 'inappropriate'입니다.\n"
        " - 'inappropriate'은 부적절성 내용을 포함하는 문장을 나타냅니다. 부적절성은 불법, 유해, 비도덕, 차별, 혐오 등과 함께 편향, 폄하, 비난 등 불쾌함을 유발하는 내용을 포함합니다.\n"
        " - 'appropriate'은 부적절성 내용이 없는 문장을 나타냅니다.\n"
        " - 출력은 'appropriate'과 'inappropriate' 중에서 1개만 생성하시오."
    )

    sentence = f"[입력 문장]\n{text}"
    user_prompt = instruction + "\n\n" + sentence

    # LLM
    if not flag_vlm:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
    # Vision-Language 
    else:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False, # Qwen3, Switches between thinking and non-thinking modes. Default is True.
        return_tensors="pt",
        # return_dict=True, # tokenize=True인 경우에 사용
    )

    return prompt
