import json
import sys
import random
import xlrd
import xlwt
sys.path.append(sys.path[0] + "/bert-utils-master/")
import extract_feature
import numpy as np

def data_process():
    data = xlrd.open_workbook(sys.path[0] + '/stastic/webSearch_data/intent_data.xlsx')
    table = data.sheet_by_index(0)
    rows = table.nrows
    
    table_data = []
    for i in range(0,rows):
        row_data = table.row_values(i)
        if "搜索" in row_data[4]:
            table_data.append(row_data)
    random.shuffle(table_data)

    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet('sheet 1')

    k = 1
    for i in range(1,int(len(table_data) * 0.8)):
        row_data = table_data[i]
        for j in range(5):
            worksheet.write(k, j, label = row_data[j])
        k += 1
    workbook.save('stastic/webSearch_data/train_data.xlsx')

    # 创建一个workbook 设置编码
    workbook1 = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet1 = workbook1.add_sheet('sheet 1')

    k = 1
    for i in range(int(len(table_data) * 0.8),len(table_data)):
        row_data = table_data[i]
        for j in range(5):
            worksheet1.write(k, j, label = row_data[j])
        k += 1
    workbook1.save('stastic/webSearch_data/test_data.xlsx')

def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if (x == zero_list).all() or (y == zero_list).all():
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

bertvector_model = None

def get_bertvector():
    global bertvector_model
    if bertvector_model is None:
        bertvector_model = extract_feature.BertVector()
        print("load bertvector_model successfull!")
    return bertvector_model

query2encode = {}
def get_query2encode():
    global query2encode
    if query2encode == {}:
        bert = get_bertvector()
        data = xlrd.open_workbook(sys.path[0] + '/stastic/webSearch_data/intent_data.xlsx')
        table = data.sheet_by_index(0)
        rows = table.nrows
        for i in range(1,rows):
            row_data = table.row_values(i)
            if row_data[0] != "" and "搜索" in row_data[4]:
                if row_data[3] != "":
                    intent = row_data[1] + "@" + row_data[2] + "@" + row_data[3]
                else:
                    intent = row_data[1] + "@" + row_data[2]
                if intent not in query2encode:
                    query2encode[intent] = [ bert.encode([row_data[0]])[0] ]
                else:
                    query2encode[intent].append(bert.encode([row_data[0]])[0])
        # with open(sys.path[0] + "/stastic/webSearch_data/query2encode.json", "w", encoding='utf-8') as fp:
        #     fp.write(json.dumps(query2encode, ensure_ascii=False, indent=4))
    return query2encode

def intent_recognization_func(sent):#意图识别
    max_score = 0
    intent = "未识别到意图"
    max_score_list = []
    bert = get_bertvector()
    query_encode = bert.encode([sent])
    query2encode = get_query2encode()
    for key in query2encode:
        score_list = []
        for q_encode in query2encode[key]:
            score_list.append(cosine_similarity(query_encode[0],q_encode))
        score_sum = 0
        score_list.sort(reverse=True)
        if len(score_list) > 5:
            score_list = score_list[:5]
        for score in score_list:
            score_sum += score
        ave_score = score_sum/len(score_list)
        if max_score < ave_score:
            max_score = ave_score
            intent = key
            max_score_list = score_list
    return max_score,intent


def test(): #测试
    data = xlrd.open_workbook(sys.path[0] + '/stastic/webSearch_data/new_data.xlsx')
    table = data.sheet_by_index(0)
    rows = table.nrows

    #创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet('sheet 1')

    for i in range(1,rows):
        try:
            j = 0
            row_data = table.row_values(i)
            intent_score,result = intent_recognization_func(row_data[0])
            worksheet.write(i, j, label = row_data[0])
            j += 1
            if intent_score > 0.92:
                if row_data[4] != "":
                    intent = row_data[2] + "@" + row_data[3] + "@" + row_data[4]
                else:
                    intent = row_data[2] + "@" + row_data[3]
                for row_intent in intent.split("@"):
                    worksheet.write(i, j, label = row_intent)
                    j += 1
            worksheet.write(i, 4, label = row_data[1])
            worksheet.write(i, 5, label = row_data[2])
            worksheet.write(i, 6, label = row_data[3])
            worksheet.write(i, 7, label = row_data[4])
        except Exception as e:
            print(e)
    workbook.save(sys.path[0] + '/stastic/webSearch_data/new_data_result_1.xlsx')

test()