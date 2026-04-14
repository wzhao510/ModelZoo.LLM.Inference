# Here is a more complex example using nested Pydantic models to handle a step-by-step math solution.

from typing import List
from pydantic import BaseModel
from openai import OpenAI

class Step(BaseModel):
    explanation: str
    output: str

class MathResponse(BaseModel):
    steps: list[Step]
    final_answer: str

client = OpenAI(base_url="http://localhost:8080/v1",api_key="-",)
model = client.models.list().data[0].id

completion = client.beta.chat.completions.parse(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful expert math tutor."},
        {"role": "user", "content": "Solve 8x + 31 = 2."},
    ],
    response_format=MathResponse,
)

message = completion.choices[0].message
print(message)
assert message.parsed
for i, step in enumerate(message.parsed.steps):
    print(f"Step #{i}:", step)
print("Answer:", message.parsed.final_answer)
if "-3.625" in message.parsed.final_answer:
    print(f"=============math_solution 测试成功=============")
else:
    print(f"=============math_solution 测试失败=============")
