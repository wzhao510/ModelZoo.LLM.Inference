import argparse
import os
import sys
import numpy as np
import pandas as pd
import time

import grpc
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
import Code.src.http.llm_pb2 as llm_pb2
import Code.src.http.llm_pb2_grpc as llm_pb2_grpc

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}


choices = ["A", "B", "C", "D"]

def run(prompts, port):
    channel = grpc.insecure_channel(f'127.0.0.1:{port}')
    llmStub = llm_pb2_grpc.LLMServiceStub(channel)
    qa_dict = {}
    req_list = llm_pb2.BatchedRequest()
    for i in range(len(prompts)):
        # print(prompts[i])
        req = llm_pb2.Request()
        req.id=i
        req.temperature=0.0
        req.generation_length=8
        req.prompt=str.encode(prompts[i])  #
        req.early_stopping=True
        req_list.req.append(req)
        if req.id not in qa_dict:
            qa_dict[req.id] = {}
            qa_dict[req.id]['question'] = prompts[i]
            qa_dict[req.id]['answer'] = ''
    
    responses = llmStub.Generation(req_list)

    # print("Answer: ")
    for response in responses:
        for element in response.rsp:
            qa_dict[element.id]['answer'] += element.generated.decode("utf-8", "ignore")

    # for id, qa in qa_dict.items():
    #     print("ID:{}".format(id))
    #     print("Question: {}".format(qa['question']))
    #     print("\n*********************************************\n")
    #     print("Answer: {}\n".format(qa['answer']))
    
    return qa_dict

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval(ntrain, subject, dev_df, test_df, port):
    cors = []

    bs = 32

    iter_num = (test_df.shape[0] + bs - 1) // bs
    for i in range(iter_num):
        # get prompt and make sure it fits
        prompts = []
        labels = []
        k = ntrain
        start = i*bs
        for j in range(bs):
            if start + j == test_df.shape[0]:
                break
            prompt_end = format_example(test_df, j+start, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            
            label = test_df.iloc[j+start, test_df.shape[1] - 1]
            prompts.append(prompt)
            labels.append(label)

        qa_dict = run(prompts, port)
        assert len(qa_dict) == len(prompts), "input error, please check"
        pred = None
        
        for id, qa in qa_dict.items(): 
            for word in qa['answer']:
                if word == " ":
                    continue
                else:
                    # print('answer', word)
                    pred = word
                    break
            if pred is None:
                # print(qa['answer'])
                pred = 'F'
            
            label = labels[id]
            cor = pred == label
            cors.append(cor)

    acc = np.mean(cors)
    cors = np.array(cors)

    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc #, all_probs


def main(data_dir, save_dir, model, port, ntrain=5):

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "results_{}".format(model))):
        os.makedirs(os.path.join(save_dir, "results_{}".format(model)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        print(f'start test {subject}')
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )
        
        cors, acc = eval(ntrain, subject, dev_df, test_df, port)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(model)] = cors
        test_df.to_csv(
            os.path.join(
                save_dir, "results_{}".format(model), "{}.csv".format(subject)
            ),
            index=None,
        )
        time.sleep(1)

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--port", "-p", type=str, default="23333")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="default_llm",
    )
    args = parser.parse_args()
    main(args.data_dir, args.save_dir, args.model, args.port, args.ntrain)
