import time
from functools import wraps
import time
from openai import OpenAI
import re
import json

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

    completion = client.chat.completions.create(
    model="model",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    )
    timestamp=completion.choices[0].message.content
    # # Do a regex to extract only the content that is valid json
    # # TODO：For whatever reason it returns null array
    # def extract_valid_json(text):
    #     pattern = r"\{(.*?)\}"
    #     matches = re.findall(pattern, text)
    #     valid_json = []
    #     for match in matches:
    #         try:
    #             json_object = json.loads(match)
    #             valid_json.append(json_object)
    #         except ValueError:
    #             continue
    #     return valid_json
    # timestamp=extract_valid_json(timestamp)
    print(f"Extracted time range:\n{timestamp}")
    return timestamp

def fake_database_lookup(timestamp):
    fake_database_result="4min15sec"
    print(f"Fake database lookup result:{fake_database_result}")
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
    completion = client.chat.completions.create(
    model="Qwen/Qwen1.5-7B-Chat-GGUF/qwen1_5-7b-chat-q4_k_m.gguf",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role":"user","content":user_prompt},
        {"role": "user", "content": auxiliary_prompt}
    ],
    temperature=0.8,
    # presence_penalty=1.0,
    # frequency_penalty=1.0,
    top_p=0.9,
    )
    response=completion.choices[0].message.content

    return response

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
    completion = client.chat.completions.create(
    model="Qwen/Qwen1.5-7B-Chat-GGUF/qwen1_5-7b-chat-q4_k_m.gguf",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3,
    presence_penalty=1.0,
    frequency_penalty=1.0,
    top_p=0.95,
    )
    response=completion.choices[0].message.content

    return response

def main():
    global client
    client = OpenAI(base_url="http://localhost:1234/v1",api_key="somekey")

    response=generate_response("How long did I stand from 10:00 p.m. to 11:00 p.m.?")
    print(response)
    response_in_chinese=generate_response_in_chinese(response)
    print(response_in_chinese)
if __name__ == "__main__":
    main()