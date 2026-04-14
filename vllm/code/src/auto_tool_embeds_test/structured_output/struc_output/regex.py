# The idea is to generate an email address, given a simple regex template.
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="-",
)
model = client.models.list().data[0].id

completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Generate an example email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: alan.turing@enigma.com\n",
        }
    ],
    extra_body={"structured_outputs": {"regex": r"\w+@\w+\.com\n"}, "stop": ["\n"]},
)
print(completion.choices[0].message.content)
if "@" in completion.choices[0].message.content:
    print(f"=================regex 测试成功=================")
else:
    print(f"=================regex 测试失败=================")
