from utils import*

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = AutoModelForCausalLM.from_pretrained("../save_load/pre_weight/gpt-sft").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("../save_load/pre_weight/gpt-sft", use_fast=False)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

input_text = "「雨降っとる」を標準語にしてください。"

text = generator(
    f"ユーザー: {input_text}\nシステム:",
    max_length=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=0,
    repetition_penalty=1.1,
    num_beams=1,
    pad_token_id=tokenizer.pad_token_id,
    num_return_sequences=1)

print(text[0]["generated_text"].split("。\nシステム:")[-1])