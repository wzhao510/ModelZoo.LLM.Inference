# The next example shows how to use the `response_format` parameter with a Pydantic model.
from pydantic import BaseModel
from enum import Enum
from openai import OpenAI

class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

json_schema = CarDescription.model_json_schema()

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
            "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "car-description",
            "schema": CarDescription.model_json_schema()
        },
    },
)
print(completion.choices[0].message.content)
if "{" in completion.choices[0].message.content and "}" in completion.choices[0].message.content:
    print(f"============response_format 测试成功============")
else:
    print(f"============response_format 测试失败============")
