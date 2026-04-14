# Can only generate positive and negative outputs.
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="-",
)
model = client.models.list().data[0].id

completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
    ],
    extra_body={"structured_outputs": {"choice": ["positive", "negative"]}},
)
print(completion.choices[0].message.content)
if completion.choices[0].message.content == "positive":
    print(f"=================choice 测试成功================")
else:
    print(f"=================choice 测试失败================")
