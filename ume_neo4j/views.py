import json
import sys
sys.path.append(sys.path[0] + "/KM_KBQA_2/src/main/")
#from service_NLP.qcls.QCLS import check_QCLS
from django.http import HttpResponse
import urllib.request
import sys
import requests, json
import os

import jieba
import re
import math
import collections
from ume_neo4j.ciLin import CilinSimilarity
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
    return -1


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
                result = json_to_str(sys.path[0] + '/stastic/webSearch_data/json_data/' + json_name)
        except Exception as e:
            result = str(e)
            print(e)
    else:
        try:
            result = json_to_str(sys.path[0] + '/stastic/webSearch_data/json_data/' + sent)
        except Exception as e:
            result = str(e)
            print(e)
    return result
            
synonym_handler =  CilinSimilarity()

def get_synonyms(word):#同义词
    synonyms = set()
    if word not in synonym_handler.vocab:
        print(word, '未被词林词林收录！')
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
    print(sent)
    result_list = web_search_func(sent,aircor)

    response = HttpResponse(json.dumps(result_list, ensure_ascii=False))
    response["Access-Control-Allow-Origin"] = "*"
    return response

def web_search_func(sent,aircor):#根据query返回页面
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
    k = 0
    if aircor == -1:
        aircor_list = ["海南航空" , "南方航空", "中国国际航空"]
    else:
        aircor_list = [aircor]
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
    countlist = []
    for i in range(len(word_list)):
        count = collections.Counter(word_list[i])
        countlist.append(count)
    text_tfidf_dict = {}
    for i, count in enumerate(countlist):
        scores = {word: tfidf(word, count, countlist) for word in seg_list_without_stopword}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        score_num = 0
        for word, score in sorted_words[:]:
            #print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
            score_num += round(score,5)
        text_tfidf_dict[subkey_to_text[i]] = score_num
    print(synonyms_result_dict)
    result_score = {}
    for word in result_dict:
        result_score[word] = result_dict[word] * 10
    for word in synonyms_result_dict:
        if word in result_score:
            result_score[word] += synonyms_result_dict[word] * 5
        else:
            result_score[word] = synonyms_result_dict[word] * 5
    for word in text_result_dict:
        if word in result_score:
            result_score[word] += text_result_dict[word]
        else:
            result_score[word] = text_result_dict[word]
    
    for word in result_score:
        if word.split("@")[-1] in text_tfidf_dict:
            result_score[word] += text_tfidf_dict[word.split("@")[-1]] * 0.5
    #print(result_score)
    a = sorted(result_score.items(), key=lambda x: x[1], reverse=True)

    result_list = []
    result_score_list = []
    if a != []:
        max_score = a[0][1]
    for result_set in a:
        air_name = ""
        if (result_set[1] > 10 and max_score > 10) or max_score < 10:
            result_score_list.append(result_set)
    #         json_name = get_target_value(result_set[0],json_data,[])
    #         for aircor_name in json_data:
    #             if get_target_value(result_set[0],json_data[aircor_name],[]) != []:
    #                 air_name = aircor_name
    #         if isinstance(json_name[0],str):
    #             result_list.append({"title":result_set[0],"source":air_name, "text":json_to_str(sys.path[0] + '/stastic/webSearch_data/json_data/' + json_name[0])})
    # #print(result_list)
    result_score_list_str = ""
    #print(result_score_list)
    for item in result_score_list:
        url_str = "/api/content_search?query=" + item[0]
        result_score_list_str = result_score_list_str + '<a href=' + url_str + '>'  + str(item)  + '</a><br/>'
    #print(result_score_list_str)
    return result_score_list_str
    