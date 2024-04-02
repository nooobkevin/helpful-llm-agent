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

model_name="Qwen/Qwen1.5-0.5B-Chat"
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
def extract_timestamp(prompt:str):
    instruction_prompt="Please extract the time range from the following text. Here's some examples:\n\n"
    query_examples=[
        {"query":"How long did Bob sit from 4:00PM to 5:00PM?","answer":"[16:00:00, 17:00:00]"},
        {"query":"How long did Bob stand from 6:00PM to 8:00PM?","answer":"[18:00:00, 20:00:00]"},
        {"query":"How long did Bob sleep from 3:00PM to 8:00PM?","answer":"[15:00:00, 20:00:00]"}
                    ]
    question_prompt=prompt
    prompt=instruction_prompt+"".join([f"Question {i+1}: {query['query']}\nAnswer {i+1}: {query['answer']}\n" for i,query in enumerate(query_examples)])
    prompt=f"{prompt}\nNow I input a new question: {question_prompt}. Please generate the time range list for this question directly, in the format of [HH:MM:SS, HH:MM:SS]. "    
    chat = [
    {"role": "user", "content": f"{prompt}"},
    ]

    tokenized_chat=tokenizer.apply_chat_template(chat,add_generation_prompt=True,tokenize=True,return_tensors='pt').to(device)
    outputs = model.generate(tokenized_chat, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id,prompt_lookup_num_tokens=10) 
    # Find the time range in the generated text formated as [HH:MM:SS, HH:MM:SS], take the last two time stamps
    time_range=re.findall(r'\d{2}:\d{2}:\d{2}',tokenizer.decode(outputs[0],skip_special_tokens=True))[-2:]
    
    print(f"Extracted time range: {time_range}")
    return time_range

def fake_database_lookup(time_stamp):
    fake_database_result="360 seconds"
    return fake_database_result

@timeit
def generate_response(user_prompt:str):
    # Extract the time range from the user prompt, and look up the database
    time_stamp=extract_timestamp(user_prompt)
    database_result=fake_database_lookup(time_stamp)
    system_prompt="You are a helpful assistant. Answer the user's question with a helpful tone. Here's an example of a helpful response:\n\n"
    auxilary_prompt=[
        {"query":"How long did I sit from 4:00PM to 5:00PM?","database_result":"300 seconds","answer":"You sat for 300 seconds from 4:00PM to 5:00PM, that is 5 minutes."},
        {"query":"How long did I stand from 6:00PM to 8:00PM?", "database_result":"720 seconds","answer":"You stood for 720 seconds from 6:00PM to 8:00PM, that is 12 minutes."},
        {"query":"How long did I sleep from 3:00PM to 8:00PM?", "database_result":"1800 seconds","answer":"You slept for 1800 seconds from 3:00PM to 8:00PM, that is 30 minutes."},
                    ]
    system_prompt=f"{system_prompt}\n".join([f"Query {i+1}: {query['query']}\nDatabase Result {i+1}: {query['database_result']}\nAnswer {i+1}: {query['answer']}\n" for i,query in enumerate(auxilary_prompt)])

    input_prompt=f"{user_prompt}\nDatabase Result: {database_result}\n"
    chat = [{"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"{input_prompt}"},
    ]
    # print(f"Chat: {tokenizer.apply_chat_template(chat,tokenize=False)}")
    
    tokenized_chat=tokenizer.apply_chat_template(chat,add_generation_prompt=True,tokenize=True,return_tensors='pt').to(device)
    outputs = model.generate(tokenized_chat, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id,prompt_lookup_num_tokens=10,temperature=0.4,do_sample=True,top_k=50,top_p=0.95) 
    result=tokenizer.decode(outputs[0],skip_special_tokens=True)
    return result.split("\n")


def main():
    global tokenizer
    global model
    tokenizer,model=load_model()
    response=generate_response("How long did I stand from 10:00 p.m. to 11:00 p.m.?")
    print(response.keys())
if __name__ == "__main__":
    main()