# Here is a simple example demonstrating how to get structured output using Pydantic models.
from pydantic import BaseModel
from openai import OpenAI

class Info(BaseModel):
    name: str
    age: int

client = OpenAI(
base_url="http://localhost:8080/v1",
api_key="-",
)
model = client.models.list().data[0].id
completion = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Cameron, I'm 28. What's my name and age?"},
    ],
    response_format=Info,
)

message = completion.choices[0].message
print(message)
assert message.parsed
print("Name:", message.parsed.name)
print("Age:", message.parsed.age)
if message.parsed.age == 28:
    print(f"===========automatic_parsing 测试成功===========")
else:
    print(f"===========automatic_parsing 测试失败===========")
