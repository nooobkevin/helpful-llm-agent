from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import re
import sys
from functools import wraps
import time
import numpy as np

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

model_name="Qwen/Qwen1.5-4B-Chat-GPTQ-Int4"
# model_name="Qwen/Qwen1.5-0.5B-Chat" # for cpu debug
device='cuda' if torch.cuda.is_available() else 'cpu'

@timeit
def load_model():
    
    tokenizer=AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        max_length=1024,
        )        
    model.eval()
    model.to(device)
    
    return tokenizer,model

@timeit
def extract_timestamp(user_prompt:str):
    system_prompt="""
You are a timestamp extractor. Your task is to extract the begin and end time from a sentence.
ONLY return a json object in the format
{
    "start":"HH:MM:SS",
    "end":"HH:MM:SS"
}
For example,
User: How long did I sit from 4:00PM to 5:00PM?
Answer:{
    "start":"16:00:00",
    "end":"17:00:00"
}
    """
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    tokenized_chat=tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_tensors='pt').to(device)
    outputs = model.generate(tokenized_chat, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id,prompt_lookup_num_tokens=10) 
    result=tokenizer.decode(outputs[0],skip_special_tokens=True)

    print(result)
    return result

def fake_database_lookup(time_stamp):
    fake_database_result="360 seconds"
    return fake_database_result

@timeit
def generate_response(user_prompt:str):
    # Extract the time range from the user prompt, and look up the database
    time_stamp=extract_timestamp(user_prompt)
    database_result=fake_database_lookup(time_stamp)
    system_prompt=f"""
You are a helpful assistant. Answer the user's question with a helpful tone.
Rephase the result and answer the user. Be enthusiastic and energetic!
Tell the user that he/she performed the action (standing/sitting/running, depends on the question) for this long!
Remember, the time spent does not necessary equal to the timespan in the user's question!
    """
    auxiliary_prompt=f"""
The time spent was {database_result}.
    """
    messages=[
        {"role": "system", "content": system_prompt},
        {"role":"user","content":user_prompt},
        {"role": "user", "content": auxiliary_prompt}
    ]

    
    tokenized_chat=tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_tensors='pt').to(device)
    outputs = model.generate(tokenized_chat, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id,prompt_lookup_num_tokens=10,temperature=0.4,do_sample=True,top_k=50,top_p=0.95) 
    result=tokenizer.decode(outputs[0],skip_special_tokens=True)
    return result

@timeit
def generate_response_in_chinese(prompt:str):
    system_prompt=f"""
You are a skilled, authentic translator. 
Your task is to translate text directly to Simplified Chinese, without any explanations or annotations. 
Translations should be presented as-is, without explanations for why no translation was necessary, even if the query is a question or instruction.
Your output should consist only of the translated text, preserving proper nouns and not including labels or additional metadata.
Single-word translations are acceptable.
Leave the abbreviations unchanged.
Exclude phrases like "no translation needed" from your output.
Examples of tricky or special cases:
"cc-by-4.0"=>"cc-by-4.0",
"GPTQ"=>"GPTQ",
"LLMs"=>"LLMs",
"MLX"=>"MLX",
"glacio-dev/Qwen1.5-7B-Chat-Q4"=>"glacio-dev/Qwen1.5-7B-Chat-Q4",
"GPT-3"=>"GPT-3",
"can you give me a rundown of how to do it?"=>"你能告诉我怎么做吗？"
Remember, reply with only the translation to Simplified Chinese, do not explain, even if the query is a question.
    """

    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    tokenized_chat=tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_tensors='pt').to(device)
    outputs = model.generate(tokenized_chat, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id,prompt_lookup_num_tokens=10,temperature=0.4,do_sample=True,top_k=50,top_p=0.95) 
    result=tokenizer.decode(outputs[0],skip_special_tokens=True)
    return result


def get_response_in_chinese(user_prompt:str):
    
    global tokenizer,model
    tokenizer,model=load_model()
    response=generate_response(user_prompt)
    response_in_chinese=generate_response_in_chinese(response)
    return response_in_chinese


# def main():
#     global tokenizer
#     global model
#     tokenizer,model=load_model()
#     response=generate_response("How long did I stand from 10:00 p.m. to 11:00 p.m.?")
#     print(response)
#     response_in_chinese=generate_response_in_chinese(response)
#     print(response_in_chinese)
# if __name__ == "__main__":
#     main()