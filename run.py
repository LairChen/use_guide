import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import TextIteratorStreamer

# model_ids: "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained('phi-2/', low_cpu_mem_usage=True, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained('phi-2/', trust_remote_code=True)

device = torch.device('cpu') # 'cuda', 'xpu', 'cpu'

# 输入示例
# inputs = tokenizer(
#     """Instruct: 用rust写一个读取文件的例子?
# Output:""" , return_tensors="pt", return_attention_mask=False).to(device)

inputs = tokenizer(
    "Instruct: 1 + 2 * 3 = ？ \n"
    "Output: "    
, return_tensors="pt", return_attention_mask=False).to(device)


# 输出
outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)

# 流式输出
# streamer = TextIteratorStreamer(tokenizer)
# generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200)

# thread = Thread(target=model.generate, kwargs=generation_kwargs)
# thread.start()
# for token in streamer:
#     print(token, end="")
# thread.join()
