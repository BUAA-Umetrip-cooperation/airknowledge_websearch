import json
import sys
import random
import xlrd
import xlwt
sys.path.append(sys.path[0] + "/bert-utils-master/")
import extract_feature
import numpy as np
import time
def data_process():
    data = xlrd.open_workbook(sys.path[0] + '/stastic/webSearch_data/new_data.xlsx')
    table = data.sheet_by_index(0)
    rows = table.nrows
    
    table_data = []
    for i in range(0,rows):
        row_data = table.row_values(i)
        #if "搜索" in row_data[4]:
        table_data.append(row_data)
    random.shuffle(table_data)

    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet('sheet 1')

    k = 1
    for i in range(1,int(len(table_data) * 0.99)):
        row_data = table_data[i]
        for j in range(len(row_data)):
            worksheet.write(k, j, label = row_data[j])
        k += 1
    workbook.save('stastic/webSearch_data/new_data_train.xlsx')

    # 创建一个workbook 设置编码
    workbook1 = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet1 = workbook1.add_sheet('sheet 1')

    k = 1
    for i in range(int(len(table_data) * 0.99),len(table_data)):
        row_data = table_data[i]
        for j in range(len(row_data)):
            worksheet1.write(k, j, label = row_data[j])
        k += 1
    workbook1.save('stastic/webSearch_data/new_data_test.xlsx')

def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    '''
    if len(x) != len(y):
        return float(0)
    zero_list = [0] * len(x)
    # if (x == zero_list).all() or (y == zero_list).all():
    #     return float(1) if x == y else float(0)

    # method 1
    # res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    # cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    start = time.time()

    #res = np.array([x[i] * y[i] for i in range(len(x))])
    cos = sum(x * y)
    end = time.time()
    print(str(round(end-start,4))+'s')

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内
    '''
    vector_a = np.mat(x)
    vector_b = np.mat(y)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

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
        # bert = get_bertvector()
        # data = xlrd.open_workbook(sys.path[0] + '/stastic/webSearch_data/new_data_train.xlsx')
        # table = data.sheet_by_index(0)
        # rows = table.nrows
        # for i in range(1,rows):
        #     try:
        #         row_data = table.row_values(i)
        #         if row_data[0] != "" and row_data[0] not in query2encode:
        #             intent = row_data[0]
        #             query2encode[intent] =  bert.encode([row_data[0]])[0].tolist() 
        #                 #f.write(intent + "\t" + str(query2encode[intent]) + "\n")
        #     except Exception as e:
        #         print(e)
        # with open(sys.path[0] + "/stastic/webSearch_data/query2encode_4.json", "w", encoding='utf-8') as fp:
        #     fp.write(json.dumps(query2encode, ensure_ascii=False, indent=4))
        with open(sys.path[0] + '/stastic/webSearch_data/query2encode_all.json','r',encoding='utf8') as fp:
            json_data = json.load(fp)
        fp.close()
        for key in json_data:
            if key not in query2encode:
                query2encode[key] = np.array(json_data[key])
                #query2encode[key] = res/(res**2).sum()**0.5
        print("载入query2encode成功！")
    return query2encode

def intent_recognization_func(sent):#意图识别
    max_score = 0
    intent = "未识别到意图"
    max_score_list = []
    bert = get_bertvector()
    query_encode = bert.encode([sent])
    query2encode = get_query2encode()
    score_dict = {}
    start = time.time()
    for key in query2encode:
        score_dict[key] = cosine_similarity(query_encode[0],query2encode[key])
    end = time.time()
    print(str(round(end-start,5))+'s')
    top_five_list = []
    #a = sorted(score_dict.items(), key=lambda x: x[1], reverse=True) #根据相似度从大到小排序
    for i in range(5):
        res_max = max(score_dict, key=lambda x: score_dict[x])
        top_five_list.append((res_max,score_dict[res_max]))
        score_dict.pop(res_max)
    return top_five_list


def test(): #测试
    data = xlrd.open_workbook(sys.path[0] + '/stastic/webSearch_data/new_data_test.xlsx')
    table = data.sheet_by_index(0)
    rows = table.nrows

    #创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet('sheet 1')
    k = 1
    query2encode = get_query2encode()
    for i in range(1,rows):
        #try:
        j = 0
        row_data = table.row_values(i)
        #start = time.time()
        if row_data[0] in query2encode:
            continue
        intent_score = intent_recognization_func(row_data[0])
        #end = time.time()
        #print(str(round(end-start,3))+'s')
        worksheet.write(k, j, label = row_data[0])
        j += 1
        for row_intent in intent_score:
            worksheet.write(k, j, label = str(row_intent))
            j += 1
        worksheet.write(k, 6, label = row_data[1])
        worksheet.write(k, 7, label = row_data[2])
        worksheet.write(k, 8, label = row_data[3])
        worksheet.write(k, 9, label = row_data[4])
        worksheet.write(k, 10, label = row_data[5])
        k += 1
        # except Exception as e:
        #     print(e)
    workbook.save(sys.path[0] + '/stastic/webSearch_data/new_data_test_result_1105.xlsx')

test()
#get_query2encode()
#data_process()