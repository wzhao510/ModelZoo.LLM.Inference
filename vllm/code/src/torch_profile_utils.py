import os
import re
import csv

# for torch profile.
def split_line_by_word_cnt(line, word_cnt_list):
    data = []

    end_idx = 0
    for i in range(len(word_cnt_list)-1):
        start_idx = word_cnt_list[i]
        end_idx = word_cnt_list[i+1]
        data_sub = line[start_idx: end_idx].strip()
        data.append(data_sub)
    data.append(line[end_idx:].strip())
    return data


def profile_to_csv(prof,  output_name="profile_data.csv"):
    data = prof.key_averages().table(sort_by="self_cuda_time_total")
    # print(data)
    lines = data.split('\n')
    
    pattern = re.compile(r'-+')
    matches = list(pattern.finditer(lines[0]))
    row_word_cnt = [match.start() for match in matches]

    # import pdb;pdb.set_trace()
    header_lines = lines[1]
    headers = split_line_by_word_cnt(header_lines, row_word_cnt)

    rows = []
    for line in lines[3:]:
        if line.startswith("---"):
            break
        rows.append(split_line_by_word_cnt(line, row_word_cnt))

    if not os.path.exists("./mx_profile/"):
        os.makedirs("./mx_profile/")
    
    with open(f"./mx_profile/{output_name}", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)
    prof.export_chrome_trace(f"./mx_profile/{output_name}.json")