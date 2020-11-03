import json
import sys
sys.path.append(sys.path[0] + "/bert-utils-master/")
#from service_NLP.qcls.QCLS import check_QCLS
from django.http import HttpResponse
import urllib.request
import sys
import requests, json
import os
import xmltodict
import jieba
import re
import math
import collections
from ume_neo4j.ciLin import CilinSimilarity

import extract_feature
import numpy as np
import time
import multiprocessing
from multiprocessing import Pool # 导入多进程中的进程池

import math
import jieba
import xlrd
import xlwt

class BM25(object):

    def __init__(self, docs):
        self.D = len(docs)
        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {} # 存储每个词及出现了该词的文档数量
        self.idf = {} # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D-v+0.5)-math.log(v+0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word]*self.f[index][word]*(self.k1+1)
                      / (self.f[index][word]+self.k1*(1-self.b+self.b*d
                                                      / self.avgdl)))
        return score

    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores


cop = re.compile("[^\u4e00-\u9fa5^，。,]") # 匹配不是中文、大小写、数字的其他字符
string1 = '@ad&*jfad张132（www）。。。'
string1 = cop.sub('', string1) #将string1中匹配到的字符替换成空字符
# word可以通过count得到，count可以通过countlist得到

# count[word]可以得到每个单词的词频， sum(count.values())得到整个句子的单词总数
def tf(word, count):
    return count[word] / sum(count.values())

# 统计的是含有该单词的句子数
def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)
 
# len(count_list)是指句子的总数，n_containing(word, count_list)是指含有该单词的句子的总数，加1是为了防止分母为0
def idf(word, count_list):
    return math.log(len(count_list) / (1 + n_containing(word, count_list)))

# 将tf和idf相乘
def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


def check_aircor(sent):#航空公司
    aircor_dict = {'南方航空':'南航','海南航空':'海航','中国国际航空':'国航'}
    for key in aircor_dict:
        if key in sent or aircor_dict[key] in sent:
            return key
    return "所有航司"


def get_target_value(key, dic, tmp_list):
    """
    :param key: 目标key值
    :param dic: JSON数据
    :param tmp_list: 用于存储获取的数据
    :return: list
    """
    if not isinstance(dic, dict) or not isinstance(tmp_list, list):  # 对传入数据进行格式校验
        return 'argv[1] not an dict or argv[-1] not an list '

    if key in dic.keys():
        tmp_list.append(dic[key])  # 传入数据存在则存入tmp_list

    for value in dic.values():  # 传入数据不符合则对其value值进行遍历
        if isinstance(value, dict):
            get_target_value(key, value, tmp_list)  # 传入数据的value值是字典，则直接调用自身
        elif isinstance(value, (list, tuple)):
            _get_value(key, value, tmp_list)  # 传入数据的value值是列表或者元组，则调用_get_value


    return tmp_list


def _get_value(key, val, tmp_list):
    for val_ in val:
        if isinstance(val_, dict):  
            get_target_value(key, val_, tmp_list)  # 传入数据的value值是字典，则调用get_target_value
        elif isinstance(val_, (list, tuple)):
            _get_value(key, val_, tmp_list)   # 传入数据的value值是列表或者元组，则调用自身


def json_to_text(json_name):#json文件处理成文本
    with open(json_name,'r',encoding='utf8') as fp:
        json_data = json.load(fp)
        text_str = json.dumps(json_data,ensure_ascii=False)
        text_str = cop.sub('', text_str) #去除其他符号
    result_text = ""
    for text in text_str.split(","):
        if text != "":
            result_text = result_text + text + "，"
    #result_text.replace("。，","。")
    #print(result_text)
    return result_text
        #print(get_target_value('#text',json_data,[]))
        #print(get_target_value('p',json_data,[]))

def json_to_str(json_name):#json文件转str，不做预处理
    with open(json_name,'r',encoding='utf8') as fp:
        json_data = json.load(fp)
    return json_data

def content_search(request): #返回文本内容
    sent = request.GET.get("query", "海南航空@旅行信息@值机服务@自助行李托运及值机")
    result_list = content_search_func(sent)

    response = HttpResponse(json.dumps(result_list, ensure_ascii=False))
    response["Access-Control-Allow-Origin"] = "*"
    return response

def json_to_xml(json_str):
    """ 传入字典字符串或字典，返回xml字符串 """
    xml_str = ''
    if type(json_str) == dict:
        dic = json_str
    else:
        dic = json.loads(json_str)
    try:
        xml_str = xmltodict.unparse(dic, encoding='utf-8')
        xml_str = xml_process(xml_str)
    except Exception as e:
        xml_str = xmltodict.unparse({'request': dic}, encoding='utf-8')  # request可根据需求修改，目的是为XML字符串提供顶层标签 , pretty=1
        xml_str = xml_process(xml_str)
    finally:
        return xml_str


# 删除xml中的多余字符
def xml_process(xml_str):
    if xml_str:
        stop_list = ["\t\t", "\n\n", "<div></div>", "<ul></ul>", "<li></li>"]
        for stop_word in stop_list:
            xml_str = xml_str.replace(stop_word, "")
    return xml_str


def content_search_func(sent): #返回文本内容
    result = ""
    if "@" in sent:
        try:
            sent_list = sent.split("@")
            with open(sys.path[0] + '/stastic/webSearch_data/all_data.json','r',encoding='utf8') as fp:
                json_data = json.load(fp)
            fp.close()
            if len(sent_list) == 3:
                sent_dict = json_data[sent_list[0]][sent_list[1]][sent_list[2]]
                result = sent_dict
            elif len(sent_list) == 4:
                sent_dict = json_data[sent_list[0]][sent_list[1]][sent_list[2]]
                json_name = ""
                for sub_dict in sent_dict:
                    if sent_list[3] in sub_dict:
                        json_name = sub_dict[sent_list[3]]
                        break
                with open(sys.path[0] + '/stastic/webSearch_data/UrlFile_map.json','r',encoding='utf8') as fp:
                    UrlFile_map_data = json.load(fp)
                fp.close()
                if json_name in UrlFile_map_data:
                    json_name = UrlFile_map_data[json_name]
                
                if "http" not in json_name:
                    json_name = sys.path[0] + '/stastic/webSearch_data/json_data/' + json_name
                    result = json_to_str(json_name)
                    result = str(json_to_xml(result))  # json转成xml后使用
                   
                else:
                    html = "<a href='{}'>{}<a>".format(json_name, sent)
                    return html
                #result = json_to_str(sys.path[0] + '/stastic/webSearch_data/json_data/' + json_name)
        except Exception as e:
            result = str(e)
            print(e)
            return "content_search_func erorr"
    else:
        try:
            result = json_to_str(sys.path[0] + '/stastic/webSearch_data/json_data/' + sent)
        except Exception as e:
            result = str(e)
            print(e)
            return "content_search_func erorr"
    return result
            
synonym_handler =  CilinSimilarity()

def get_synonyms(word):#同义词
    synonyms = set()
    if word not in synonym_handler.vocab:
        pass
        #print(word, '未被词林词林收录！')
    else:
        codes = synonym_handler.word_code[word]
        for code in codes:
            key = synonym_handler.code_word[code]
            synonyms.update(key)
        if word in synonyms:
            synonyms.remove(word)

    return list(synonyms)

def web_search(request): #query检索
    sent = request.GET.get("query", "托运行李")
    aircor = check_aircor(sent)
    aircor_dict = {'南方航空':'南航','海南航空':'海航','中国国际航空':'国航'}
    for key in aircor_dict:
        if key in sent:
            sent = sent.replace(key,"")
        if aircor_dict[key] in sent:
            sent = sent.replace(aircor_dict[key],"")
    
    result_list = web_search_func(sent,aircor)

    response = HttpResponse(json.dumps(result_list, ensure_ascii=False))
    response["Access-Control-Allow-Origin"] = "*"
    return response

def get_query_description(query,text):#获得title下关于query的description
    #print(query,text)
    s = BM25(text)
    score = s.simall(query)
    result_score = {}
    for i in range(len(text)):
        result_score[text[i]] = score[i]
    a = sorted(result_score.items(), key=lambda x: x[1], reverse=True)
    result_str = ""
    str_index = text.index(a[0][0])
    result_str = "".join(text[str_index:])
    high_line = []
    if len(result_str) < 50:
        for word in query:
            if word in result_str:
                high_line.append(word)
        return result_str,high_line
    else:
        for word in query:
            if word in result_str[:50]:
                high_line.append(word)
        return result_str[:50],high_line


def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])

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

def intent_recognization(request): #意图识别
    sent = request.GET.get("sent", "托运行李")
    aircor = request.GET.get("aircor", "所有航司")

    result_list = intent_recognization_func(sent,aircor)

    response = HttpResponse(json.dumps(result_list, ensure_ascii=False))
    response["Access-Control-Allow-Origin"] = "*"
    return response

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
        #print(cosine_similarity(query_encode[0],query2encode[key][0]))
        if max_score < ave_score:
            max_score = ave_score
            intent = key
            max_score_list = score_list
    #print(max_score,intent,max_score_list)
    return max_score,intent
    # bert = get_bertvector()
    # v1 = bert.encode(["可以提前选航班座位吗"])
    # v2 = bert.encode(["哪里可以提前选座位"])
    # #print(v1,v2)
    # print(cosine_similarity(v1[0],v2[0]))

def str_match(request): #字符串匹配
    sent = request.GET.get("sent", "托运行李")
    aircor = request.GET.get("aircor", "所有航司")

    result_list = str_match_func(sent,aircor)

    response = HttpResponse(json.dumps(result_list, ensure_ascii=False))
    response["Access-Control-Allow-Origin"] = "*"
    return response        

def str_match_func(sent,aircor_name):#字符串匹配
    # 读取json文件内容,返回字典格式
    with open(sys.path[0] + '/stastic/webSearch_data/all_data.json','r',encoding='utf8') as fp:
        json_data = json.load(fp)
    fp.close()
    with open(sys.path[0] + '/stastic/webSearch_data/UrlFile_map.json','r',encoding='utf8') as fp:
        UrlFile_map_data = json.load(fp)
    fp.close()
    stopword_list = []
    file = open(sys.path[0] + "/stastic/webSearch_data/stopwords-master/baidu_stopwords.txt") #加载停用词表
    for line in file:
        if line.strip("\n") not in stopword_list:
            stopword_list.append(line.strip("\n"))
    file.close()
    seg_list = jieba.lcut(sent,cut_all = False)#精确分词
    #seg_list2 = jieba.lcut_for_search(sent)
    seg_list_without_stopword = []#去停用词
    for word in seg_list:
        if word not in stopword_list:
            seg_list_without_stopword.append(word)
    
    result_dict = {}
    text_result_dict = {}
    synonyms_result_dict = {}
    word_list = []
    subkey_to_text = {}
    doc = {}#BM25
    k = 0
    print("aircor的值为:",aircor_name)
    if aircor_name == "所有航司":
        aircor_list = ["海南航空" , "南方航空", "中国国际航空"]
    else:
        aircor_list = [aircor_name]
    for aircor in aircor_list:
        for kk in json_data[aircor]:
            for key in json_data[aircor][kk]:
                if aircor + "@" + kk + "@" + key in result_dict:
                    continue
                for word in seg_list_without_stopword:
                    synonyms_list = get_synonyms(word)
                    if word in key: #标题
                        if aircor + "@" + kk + "@" + key not in result_dict:
                            result_dict[aircor + "@" + kk + "@" + key] = 1
                        else:
                            result_dict[aircor + "@" + kk + "@" + key] += 1
                    else:#该词不在标题中，查看同义词是否在标题
                        for synonyms_word in synonyms_list:
                            if synonyms_word in key:
                                if aircor + "@" + kk + "@" + key not in synonyms_result_dict:
                                    synonyms_result_dict[aircor + "@" + kk + "@" + key] = 1
                                else:
                                    synonyms_result_dict[aircor + "@" + kk + "@" + key] += 1
                for sub_dict in json_data[aircor][kk][key]:
                    for sub_key in sub_dict:
                        if aircor + "@" + kk + "@" + key + "@" + sub_key in result_dict:
                            continue
                        sub_key_text = ""
                        try:
                            if sub_dict[sub_key] in UrlFile_map_data:
                                sub_dict[sub_key] = UrlFile_map_data[sub_dict[sub_key]]
                            sub_key_text = json_to_text(sys.path[0] + '/stastic/webSearch_data/json_data/' + sub_dict[sub_key])
                            if sub_key_text != "":
                                word_list.append(jieba.lcut(sub_key_text,cut_all = False))#对文本进行分词
                                subkey_to_text[k] = sub_key
                                k += 1
                                doc[aircor + "@" + kk + "@" + key + "@" + sub_key] = jieba.lcut(sub_key_text,cut_all = False)
                        except Exception as e:
                                #print(e)
                                pass
                        for word in seg_list_without_stopword:
                            synonyms_list = get_synonyms(word)
                            if word in sub_key:
                                if aircor + "@" + kk + "@" + key + "@" + sub_key not in result_dict:
                                    result_dict[aircor + "@" + kk + "@" + key + "@" + sub_key] = 1
                                else:
                                    result_dict[aircor + "@" + kk + "@" + key + "@" + sub_key] += 1
                            else:#该词不在标题中，查看同义词是否在标题
                                for synonyms_word in synonyms_list:
                                    if synonyms_word in sub_key:
                                        if aircor + "@" + kk + "@" + key + "@" + sub_key not in synonyms_result_dict:
                                            synonyms_result_dict[aircor + "@" + kk + "@" + key + "@" + sub_key] = 1
                                        else:
                                            synonyms_result_dict[aircor + "@" + kk + "@" + key + "@" + sub_key] += 1
                                
                            if word in sub_key_text:
                                if aircor + "@" + kk + "@" + key + "@" + sub_key not in text_result_dict:
                                    text_result_dict[aircor + "@" + kk + "@" + key + "@" + sub_key] = 1
                                else:
                                    text_result_dict[aircor + "@" + kk + "@" + key + "@" + sub_key] += 1
    
    doc_title = []
    doc_content = []
    for key in doc:
        doc_title.append(key)
        doc_content.append(doc[key])
    s = BM25(doc_content)
    BM25_score = s.simall(seg_list_without_stopword)
    BM25_score_dict = {}
    for i in range(len(doc_title)):
        BM25_score_dict[doc_title[i]] = BM25_score[i]
    
    result_score = {}
    for word in result_dict:
        result_score[word] = result_dict[word] * 10
    for word in synonyms_result_dict:
        if word in result_score:
            result_score[word] += synonyms_result_dict[word] * 5
        else:
            result_score[word] = synonyms_result_dict[word] * 5
    for word in BM25_score_dict:
        if word in result_score:
            result_score[word] += BM25_score_dict[word]
        else:
            result_score[word] = BM25_score_dict[word]
    
    return result_score,doc,seg_list_without_stopword

def get_intent_rec_result(sent,aircor):
    github_url = "http://127.0.0.1:12345/api/intent_recognization?sent=" + sent + "&aircor=" + str(aircor)
    r = requests.get(github_url)
    result_list = []
    for item in r.json():
        result_list.append(item)
    #print("结果:",result_list)
    return result_list

def get_str_match_result(sent,aircor):
    github_url = "http://127.0.0.1:12345/api/str_match?sent=" + sent + "&aircor=" + str(aircor)
    r = requests.get(github_url)
    result_list = []
    for item in r.json():
        result_list.append(item)
    #print("结果:",result_list)
    return result_list

def web_search_func(sent,aircor_name):#根据query返回页面

    start = time.time()
    # pool = Pool(processes=2) #多进程
    # q = []
    # q.append(pool.apply_async(get_intent_rec_result, args=(str(sent),aircor_name)))
    # q.append(pool.apply_async(get_str_match_result, args=(str(sent), aircor_name)))
    # for item in q:
    #     r = item.get()
    #     #print("r:",r)
    #     if len(r) == 2:
    #         intent_score,intent = r[0],r[1]
    #     else:
    #         result_score,doc,seg_list_without_stopword = r[0],r[1],r[2]
    # pool.close()
    # pool.join()
    # r = get_intent_rec_result(sent,aircor_name)
    # intent_score,intent = r[0],r[1]
    # r = get_str_match_result(sent,aircor_name)
    # result_score,doc,seg_list_without_stopword = r[0],r[1],r[2]

    intent_score,intent = intent_recognization_func(sent)#意图识别
    result_score,doc,seg_list_without_stopword = str_match_func(sent,aircor)#字符串匹配
    end = time.time()
    print(str(round(end-start,3))+'s')

    final_result_list = []
    #final_result_tag_list = []
    if intent_score > 0.92:#意图识别的结果超过阈值
        with open(sys.path[0] + '/stastic/webSearch_data/web2intent.json','r',encoding='utf8') as fp:
            web2intent_data = json.load(fp)
        fp.close()
        web_title = "未找到相关页面"
        for key in web2intent_data:
            if web2intent_data[key] == intent:
                web_title = key
                break
        if web_title != "未找到相关页面":
            with open(sys.path[0] + '/stastic/webSearch_data/all_data.json','r',encoding='utf8') as fp:
                json_data = json.load(fp)
            fp.close()
            if aircor_name == "所有航司":
                aircor_list = ["海南航空" , "南方航空", "中国国际航空"]
            else:
                aircor_list = [aircor_name]
            for aircor in aircor_list:
                for kk in json_data[aircor]:
                    for key in json_data[aircor][kk]:
                        for sub_dict in json_data[aircor][kk][key]:
                            for sub_key in sub_dict:
                                if sub_key == web_title:
                                    web_title = aircor + "@" + kk + "@" + key + "@" + sub_key
                                    break
            if len(web_title.split("@")) <= 1:
                pass
            else:
                if web_title in result_score and result_score[web_title] < 25:
                    if intent_score > 0.96:
                        result_score[web_title] = 35
                    else:
                        result_score[web_title] = 25
    a = sorted(result_score.items(), key=lambda x: x[1], reverse=True)

    result_score_list = []
    if a != []:
        max_score = a[0][1]
    for result_set in a:
        air_name = ""
        #if result_set[1] > 3 :
        if max_score >= 20:
        #if (result_set[1] > 10 and max_score > 10) or max_score < 10:
            if result_set[1] >= 20:
                result_score_list.append(result_set)
        elif max_score >= 10 and max_score < 20:
            if result_set[1] >= 10:
                result_score_list.append(result_set)
        else:
            if result_set[1] >= 5:
                result_score_list.append(result_set)

            # url_str = "/api/content_search?query=" + web_title
            # if web_title in doc:
            #     item_text = "".join(doc[web_title])
            #     item_text_list = []
            #     for t in item_text.split("。"):
            #         item_text_list.append(t.strip("，"))
            #     description,high_line = get_query_description(seg_list_without_stopword,item_text_list)
            # else:
            #     description,high_line = "",[]
            # result_score_list_str =  '<a href=' + url_str + '>'  + str(web_title.split("@")[-1])  +  '</a>'
            # final_result_list.append({"title":result_score_list_str,"source":web_title.split("@")[0],"score":intent_score,"content":description + "...","high_line":high_line})
            # final_result_tag_list.append(web_title)
    
    for item in result_score_list:
        result_score_list_str = ""
        url_str = "/api/content_search?query=" + item[0]
        # text_score = result_dict[item[0]] * 10 if item[0] in result_dict else 0
        # synonyms_score = synonyms_result_dict[item[0]] * 5 if item[0] in synonyms_result_dict else 0
        # BM25_score = BM25_score_dict[item[0]] if item[0] in BM25_score_dict else 0
        if item[0] in doc:
            item_text = "".join(doc[item[0]])
            item_text_list = []
            for t in item_text.split("。"):
                item_text_list.append(t.strip("，"))
            description,high_line = get_query_description(seg_list_without_stopword,item_text_list)
        else:
            description,high_line = "",[]
        #every_score_str = "text_score:" + str(text_score) + ",synonyms_score: " + str(synonyms_score) + ",BM25_score:" + str(BM25_score)
        description_str = ",description:" + description + "high_line:" + str(high_line)
        result_score_list_str =  '<a href=' + url_str + '>'  + str(item[0].split("@")[-1])  +  '</a>'
        final_result_list.append({"title":result_score_list_str,"source":item[0].split("@")[0],"score":item[1],"content":description + "...","high_line":high_line})
    #print(result_score_list_str)
    return final_result_list
    
def test(request): #测试
    sent = request.GET.get("query", "托运行李")
    data = xlrd.open_workbook(sys.path[0] + '/stastic/webSearch_data/new_data.xlsx')
    table = data.sheet_by_index(0)
    rows = table.nrows
    right_num ,error_num = 0,0

    #创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet('sheet 1')

    for i in range(1,rows):
        print(i)
        if i == 10000:
            break
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
            if intent == result:
                right_num += 1
            else:
                error_num += 1
            for row_intent in intent.split("@"):
                worksheet.write(i, j, label = row_intent)
                j += 1
        worksheet.write(i, 4, label = row_data[1])
        worksheet.write(i, 5, label = row_data[2])
        worksheet.write(i, 6, label = row_data[3])
        worksheet.write(i, 7, label = row_data[4])
    workbook.save('new_data_result_1.xlsx')

    result_list = {"right_num":right_num,"error_num":error_num,"all_num":rows}
    response = HttpResponse(json.dumps(result_list, ensure_ascii=False))
    response["Access-Control-Allow-Origin"] = "*"
    return response