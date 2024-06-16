from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 加载保存的模型和分词器
output_dir = "./gpt2_jin_yong"
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model = GPT2LMHeadModel.from_pretrained(output_dir)

# Function to generate text using the trained model with additional parameters
# def generate_text_transformer(seed_text, next_words, model, tokenizer, temperature=0.7, top_k=50, top_p=0.95):
#     input_ids = tokenizer.encode(seed_text, return_tensors='pt')
#     output = model.generate(
#         input_ids, 
#         max_length=next_words + len(input_ids[0]), 
#         num_return_sequences=1, 
#         temperature=temperature, 
#         top_k=top_k, 
#         top_p=top_p,
#         do_sample=True
#     )
#     return tokenizer.decode(output[0], skip_special_tokens=True)
# 定义生成文本的函数
def generate_text_transformer(seed_text, additional_words, model, tokenizer, temperature=0.7, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=len(input_ids[0]) + additional_words,
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id  # 确保pad_token_id设置为eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)



# Generate a sample text with improved parameters
# sample_text = "张无忌快步走近山脚，正要上峰，忽见山道旁中白光微闪，有人执着兵刃埋伏。他急忙停步，只过得片刻，见树丛中先后窜出四人，三前一后，齐向峰顶奔去。"
sample_text = "幸好，十八年之约即将到来，六怪便带着郭靖来到江南。郭靖奉师命先行，遇到装扮成男乞丐的黄蓉，并且对她悉心照顾，两人结伴而行，一直来到金国的中都。"
# sample_text = "乔峰来姑苏，本是找慕容复查清丐帮副帮主马大元被他自己的成名绝技所杀一事，谁知帮内突生大变，他被指证为契丹人。为解开自己的身世之谜，他北上少室山，找自己的养父乔三槐和恩师玄苦，可二人已遇害身亡，目击之人皆认为是乔峰所为。"
# print(generate_text_transformer(sample_text, 200, model, tokenizer))

# !尝试不同的参数组合

generated_text1 = generate_text_transformer(sample_text, 300, model, tokenizer, temperature=0.7, top_k=30, top_p=0.85)
generated_text2 = generate_text_transformer(sample_text, 300, model, tokenizer, temperature=0.8, top_k=40, top_p=0.9)
generated_text3 = generate_text_transformer(sample_text, 300, model, tokenizer, temperature=1.0, top_k=50, top_p=0.95)
print('generated_text1:',generated_text1)
print('generated_text2:',generated_text2)
print('generated_text3:',generated_text3)
