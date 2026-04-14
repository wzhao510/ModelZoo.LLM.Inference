# Finally we have the `grammar` option, which is probably the most
# difficult to use, but it´s really powerful. It allows us to define complete
# languages like SQL queries. It works by using a context free EBNF grammar.
# As an example, we can use to define a specific format of simplified SQL queries.
from openai import OpenAI

client = OpenAI(
base_url="http://localhost:8080/v1",
api_key="-",
)
model = client.models.list().data[0].id

simplified_sql_grammar = """
    root ::= select_statement

    select_statement ::= "SELECT " column " from " table " where " condition

    column ::= "col_1 " | "col_2 "

    table ::= "table_1 " | "table_2 "

    condition ::= column "= " number

    number ::= "1 " | "2 "
"""

completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Generate an SQL query to show the 'username' and 'email' from the 'users' table.",
        }
    ],
    extra_body={"structured_outputs": {"grammar": simplified_sql_grammar}},
)
print(completion.choices[0].message.content)
if "SELECT" in completion.choices[0].message.content:
    print(f"================grammar 测试成功================")
else:
    print(f"================grammar 测试失败================")
