import json, time
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertModel, BertTokenizer
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# import paddle
# import paddle.nn.functional as F
import unicodedata
from pyhanlp import *
from torch_geometric.nn import RGCNConv
from gcn import *
from graphModule import *
from einops import rearrange
from config import args
from biaffine import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
# BERT_PATH = "./SpanBERT/Spanbert-base-cased"
# BERT_PATH = "./chinese_roberta_wwm_ext_pytorch"
BERT_PATH = "./bert"
maxlen = 256  ####256


def load_data(filename):
    D = []
    with open(filename) as data_file:
        data = data_file.read()
        # print(data)
        data = json.loads(data)
        for item in data:
            d = {'text': item['text'], 'triple_list': []}
            for sub_item in item['triple_list']:
                d['triple_list'].append(
                    (sub_item[0], sub_item[1], sub_item[2])
                )
            D.append(d)

    return D


# 加载数据集
train_data = load_data('./data/CMED/train_triples.json')
valid_data = load_data('./data/CMED/dev_triples.json')


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


train_data_new = []  # 创建新的训练集，把结束位置超过250的文本去除，可见并没有去除多少
for data in tqdm(train_data):
    # print (data)
    flag = 1
    for s, p, o in data['triple_list']:
        s_begin = search(s, data['text'])
        o_begin = search(o, data['text'])
        if s_begin == -1 or o_begin == -1 or s_begin + len(s) > 256 or o_begin + len(o) > 256:
            flag = 0
            break
    if flag == 1:
        train_data_new.append(data)
print("去除大于250的文本:\t", len(train_data_new))

# 读取schema
'''
with open('RE/data/schema.json', encoding='utf-8') as f:
    id2predicate, predicate2id, n = {}, {}, 0
    predicate2type = {}
    for l in f:
        l = json.loads(l)
        predicate2type[l['predicate']] = (l['subject_type'], l['object_type'])
        for k, _ in sorted(l['object_type'].items()):
            key = l['predicate'] + '_' + k
            id2predicate[n] = key
            predicate2id[key] = n
            n += 1
print(len(predicate2id))
'''

with open('./data/CMED/rel2id.json', encoding='utf-8') as f:
    # id2predicate, predicate2id, n = {}, {}, 0
    l = json.load(f)
    id2predicate = l[0]
    predicate2id = l[1]
print("关系类型数量:\t", len(predicate2id))


class OurTokenizer(BertTokenizer):
    def tokenize(self, text):
        R = []
        for c in text:
            if c in self.vocab:
                R.append(c)
            elif self._is_whitespace(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R

    def _is_whitespace(self, char):
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False


# 初始化分词器
tokenizer = OurTokenizer(vocab_file="./chinese_roberta_wwm_ext_pytorch/vocab.txt")


######依存句法树+分词
def seg_pos(text):
    head, seg_word, Dep_rel, str_le = [], [], [], []
    # tree = HanLP.parseDependency(text)
    parser = JClass('com.hankcs.hanlp.dependency.nnparser.NeuralNetworkDependencyParser')()
    parser.enableDeprelTranslator(False)
    tree = parser.parse(text)
    for word in tree.iterator():  # 通过dir()可以查看sentence的方法
        head.append(word.HEAD.ID)
        for i in word.LEMMA.split():
            str_le.append(i)
        seg_word.append(word.LEMMA)
        Dep_rel.append(word.DEPREL)
    return head, seg_word, Dep_rel, str_le


def out_list_word(seg_word):
    temp = ""
    for word in seg_word:
        temp += " " + word
        text_out = temp.lstrip(" ")
    return text_out


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab.keys() else 0 for t in tokens]
    return ids


def vocab_json():
    vocab_out = json.load(open("./vacab.json"))
    return vocab_out


def dep_json():
    dep_out = json.load(open("./dep.json"))
    return dep_out


class TorchDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        t = self.data[i]
        # print ('t!!!',t) ######{'text': '齐志江，男，汉族，中共党员，大学学历', 'triple_list': [('齐志江', '民族', '汉族')]}

        x = tokenizer.tokenize(t['text'])
        # print (x)
        x = ["[CLS]"] + x + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(x)
        seg_ids = [0] * len(token_ids)
        assert len(token_ids) == len(t['text']) + 2
        spoes = {}
        for s, p, o in t['triple_list']:
            s = tokenizer.tokenize(s)
            s = tokenizer.convert_tokens_to_ids(s)
            p = predicate2id[p]
            o = tokenizer.tokenize(o)
            o = tokenizer.convert_tokens_to_ids(o)
            s_idx = search(s, token_ids)
            o_idx = search(o, token_ids)

            if s_idx != -1 and o_idx != -1:
                s = (s_idx, s_idx + len(s) - 1)
                o = (o_idx, o_idx + len(o) - 1, p)  # 同时预测o和p
                if s not in spoes:
                    spoes[s] = []
                spoes[s].append(o)
        # print(spoes) {(2, 5): [(13, 15, 31), (19, 21, 38), (29, 31, 45)]}

        if spoes:
            sub_labels = np.zeros((len(token_ids), 2))
            # print (sub_labels)
            for s in spoes:
                # print (s) #(2, 5)
                # print (sub_labels)
                # print(s[0])
                sub_labels[s[0], 0] = 1
                sub_labels[s[1], 1] = 1
            # 随机选一个subject
            start, end = np.array(list(spoes.keys())).T
            start = np.random.choice(start)
            # print (start)
            end = sorted(end[end >= start])[0]
            sub_ids = (start, end)
            obj_labels = np.zeros((len(token_ids), len(predicate2id), 2))
            for o in spoes.get(sub_ids, []):
                # print (o)
                obj_labels[o[0], o[2], 0] = 1
                obj_labels[o[1], o[2], 1] = 1

        token_ids = self.sequence_padding(token_ids, maxlen=maxlen)
        seg_ids = self.sequence_padding(seg_ids, maxlen=maxlen)
        sub_labels = self.sequence_padding(sub_labels, maxlen=maxlen, padding=np.zeros(2))
        sub_ids = np.array(sub_ids)
        obj_labels = self.sequence_padding(obj_labels, maxlen=maxlen,
                                           padding=np.zeros((len(predicate2id), 2)))

        return (torch.LongTensor(token_ids), torch.LongTensor(seg_ids), torch.LongTensor(sub_ids),
                torch.LongTensor(sub_labels), torch.LongTensor(obj_labels))

    def __len__(self):
        data_len = len(self.data)
        return data_len

    def sequence_padding(self, x, maxlen, padding=0):
        output = np.concatenate([x, [padding] * (maxlen - len(x))]) if len(x) < maxlen else np.array(x[:maxlen])
        return output


train_dataset = TorchDataset(train_data_new)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch1, shuffle=True, drop_last=True)


# for i, x in enumerate(train_loader):
#     print([_.shape for _ in x])
#     if i == 10:
#         break
class GRUnet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        """
        vocab_size: 词典长度，也就是嵌入矩阵的行数
        embedding_dim: 词向量的维度，也就是嵌入矩阵的列数，也是W的列数，也是输入GRU的x_t的维度
        hidden_dim: GRU神经元的个数，也就是W的行数
        layer_dim: GRU的层数
        output_dim: 隐藏层输出的维度
        """
        super(GRUnet, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # GRU + 全连接
        self.gru = nn.GRU(embedding_dim, hidden_dim, layer_dim,
                          batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x : [bacth, time_step, vocab_size]
        embeds = self.embedding(x)
        # print(embeds.shape)
        # embeds : [batch, time_step, embedding_dim]
        r_out, h_n = self.gru(embeds, None)
        # print (r_out.shape)
        # r_out : [batch, time_step, hidden_dim]
        # out = self.fc1(r_out[:, -1, :])
        out = self.fc1(r_out)
        # out : [batch, time_step, output_dim]
        return out


class GCN(nn.Module):
    def __init__(self, hidden_size=768):
        super(GCN, self).__init__()
        self.hidden_size = hidden_size
        # self.fc = nn.Linear(self.hidden_size, self.hidden_size // 2)

    def forward(self, x, adj, is_relu=True):
        out = x

        # Make permutations for matrix multiplication
        # Assuming batch_first = False
        # print (out.shape)
        # out = out.permute(1, 0, 2) # to: batch, seq_len, hidden
        # adj = adj.permute(2, 0, 1) # to: batch, seq_len, seq_len

        out = torch.bmm(adj, out)  # .permute(1, 0, 2) # to: seq_len, batch, hidden

        if is_relu == True:
            out = F.relu(out)

        return out



class RGCN(torch.nn.Module):
    def __init__(self,in_channels,hideden_channels,out_channels,n_layers=2,dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.relu = F.relu
        self.dropout = dropout
        self.convs.append(RGCNConv(in_channels,hideden_channels,num_relations=24,num_bases=1))
        for i in range(n_layers-2):
            self.convs.append(RGCNConv(hideden_channels,hideden_channels,num_relations=24,num_bases=1))
            self.norms.append(torch.nn.BatchNormld(hideden_channels))
        self.convs.append(RGCNConv(hideden_channels,out_channels,num_relations=24,num_bases=1))
    def forward(self, x, edge_index=2561,edge_type=24):
        for conv ,norm in zip(self.convs, self.norms):
            x = norm(conv(x,2561,24))
            x = F.relu(x)
            x = F.dropout(x,p=self.dropout,training=self.training)
        return x

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)  # [bs, maxlen, 1]
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Attention2(nn.Module):
    """
    1.输入 [batch_size,time_step,hidden_dim] -> Linear、Tanh
    2.[batch_size,time_step,hidden_dim] -> transpose
    3.[batch_size,hidden_dim,time_step] -> Softmax
    4.[batch_size,hidden_dim,time_step] -> mean
    5.[batch_size,time_step] -> unsqueeze
    5.[batch_size,1,time_step] -> expand
    6.[batch_size,hidden_dim,time_step] -> transpose
    7.[batch_size,time_step,hidden_dim]
    """

    def __init__(self, hidden_dim):
        super(Attention2, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features, mean=True):
        batch_size, time_step, hidden_dim = features.size()
        # weight = nn.Tanh()(self.dense(features))
        weight = nn.ReLU()(self.dense(features))

        # mask给负无穷使得权重为0
        mask_idx = torch.sign(torch.abs(features).sum(dim=-1))
        mask_idx = mask_idx.unsqueeze(-1).expand(batch_size, time_step, hidden_dim)
        paddings = torch.ones_like(mask_idx) * (-2 ** 32 + 1)
        weight = torch.where(torch.eq(mask_idx, 1), weight, paddings)

        weight = weight.transpose(2, 1)
        # weight = nn.Softmax(dim=2)(weight)
        # weight = nn.Sigmoid(weight)
        if mean:
            weight = weight.mean(dim=1)
            weight = weight.unsqueeze(1)
            weight = weight.expand(batch_size, hidden_dim, time_step)
        weight = weight.transpose(2, 1)
        features_attention = weight * features

        return features_attention


class KeyValueMemoryNetwork(nn.Module):
    def __init__(self, vocab_size, feature_vocab_size, emb_size):
        super(KeyValueMemoryNetwork, self).__init__()
        self.key_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.value_embedding = nn.Embedding(feature_vocab_size, emb_size, padding_idx=0)
        self.scale = np.power(emb_size, 0.5)

    def forward(self, key_embed, value_embed, hidden, mask_matrix):
        # key_embed = self.key_embedding(key_seq)
        # print (key_embed.shape)
        # value_embed = self.value_embedding(value_seq)
        # print (value_embed.shape)
        # hidden = self.key_embedding(hidden)
        u = torch.bmm(hidden.float(), key_embed.transpose(1, 2))
        u = u / self.scale
        exp_u = torch.exp(u)
        # print ('exp_u',exp_u.shape)
        delta_exp_u = torch.mul(exp_u.float(), mask_matrix.float())
        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)
        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)
        # print ('exp_u',p.shape)(9,256,256)
        # embedding_val = value_embed.permute(3, 0, 1, 2)
        o = torch.mul(p.float(), value_embed.float())
        # print (o.shape)
        # o = o.permute(1, 2, 3, 0)
        # o = torch.sum(o, 2)

        # aspect_len = (o != 0).sum(dim=1)
        # o = o.float().sum(dim=1)
        # avg_o = torch.div(o, aspect_len)
        return o  # avg_o.type_as(hidden)



class REModel(nn.Module):
    def __init__(self):
        super(REModel, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.linear = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        self.sub_output = nn.Linear(768, 2)
        self.suopand = nn.Linear(1024, 768)
        self.cat_output = nn.Linear(1024, 768)
        self.obj_output = nn.Linear(768, len(predicate2id) * 2)

        self.sub_pos_emb = nn.Embedding(256, 768)  # subject位置embedding
        self.layernorm = BertLayerNorm(768, eps=1e-12)
        # self.GCN_model = GCNClassifier(opt, emb_matrix=None)

        self.GRU = GRUnet(23923, 768, 1024, 6, 768)
        # self.CRF_S = CRF_S(768, 16, if_bias=True)
        # self.LSTM_CRF = LSTM_CRF(23922, 16, 768, 768, 1, 0.5, large_CRF=True)

        self.biaffine = BiaffineTagger(768, 2)

        # self.GCN = GCN(hidden_size=768)
        self.attention2 = Attention2(hidden_dim=768)
        self.gcu1 = GraphConv1(batch=args.batch1, h=[16, 32, 64, 128, 256], w=[16, 32, 64, 128, 256], d=[768, 512],
                               V=[2, 4, 8, 32], outfeatures=[256, 128])
        # self.gcu2 = GraphConv2(batch = args.batch2, h=[16,32,64,128,256], w=[16,32,64,128,256], d=[768,512], V=[2,4,8,32],outfeatures=[256,128])
        self.cov = nn.Conv2d(768, 768, 1)
        self.GCN_model = GCNClassifier(opt, emb_matrix=None)
        self.emb = nn.Embedding(23923, 768)
        self.emb1 = nn.Embedding(37, 256)
        self.keyvalue = KeyValueMemoryNetwork(23923, 23923, 768)
        # self.apnb = APNB(in_channels=768, out_channels=768, key_channels=256, value_channels=256,dropout=0.05, sizes=([1]))

    def forward(self, token_ids, seg_ids, sub_ids=None):
        out, _ = self.bert(token_ids, token_type_ids=seg_ids,
                           output_all_encoded_layers=False)  # [batch_size, maxlen, size]
        # print ("1",out.shape)
        out = self.attention2(out)
        # print("1", out.shape)
        sub_preds = self.sub_output(out)  # [batch_size, maxlen, 2]
        sub_preds = torch.sigmoid(sub_preds)
        # sub_preds = sub_preds ** 2

        if sub_ids is None:
            return sub_preds

        # print(sub_ids)
        # print(sub_ids[:, :1])
        # 融入subject特征信息
        sub_pos_start = self.sub_pos_emb(sub_ids[:, :1])  # 取主实体首位置
        sub_pos_end = self.sub_pos_emb(sub_ids[:, 1:])  # [batch_size, 1, size] #取主实体尾位置

        # print(sub_pos_start)

        sub_id1 = sub_ids[:, :1].unsqueeze(-1).repeat(1, 1, out.shape[-1])  # subject开始的位置id 重复字编码次数
        # print (sub_id1)
        sub_id2 = sub_ids[:, 1:].unsqueeze(-1).repeat(1, 1, out.shape[-1])  # [batch_size, 1, size]
        sub_start = torch.gather(out, 1, sub_id1)  # 按照sub_id1位置索引去找bert编码后的值，在列维度进行索引
        # print(sub_start.shape)
        sub_end = torch.gather(out, 1, sub_id2)  # [batch_size, 1, size]

        sub_start = sub_pos_start + sub_start  # 位置编码向量+bert字编码向量
        sub_end = sub_pos_end + sub_end
        out1 = out + sub_start + sub_end

        out1 = torch.reshape(out1, (-1, 16, 16, 768))
        # print ('out1:',out1.shape)
        out1 = out1.permute(0, 3, 1, 2)
        # print(out1.shape)
        # out1 = HGT(in_channels=1, hidden_channels=5, out_channels=2, n_layers=2, n_heads=3)(out1)

        out1 = RGCN(in_channels=1, hideden_channels=5, out_channels=2, n_layers=2, dropout=0.5)(out1)
        # print(1)out1 = RGCN(in_channels=1, hideden_channels=5, out_channels=2, n_layers=2, dropout=0.5)(out1)
        # print(out1.shape)
        # print(1)
        # print(out1.shape)
        # if out1.shape[0] == args.batch1:
        #     out1 = self.gcu1(out1)
        #     # word_re_embed,_ = self.LSTM_CRF(inputs[0],hidden=None,t = True)
        # else:
        #     out1 = GraphConv2(batch=out1.shape[0], h=[16, 32, 64, 128, 256], w=[16, 32, 64, 128, 256], d=[768, 512],
        #                       V=[2, 4, 8, 32], outfeatures=[256, 128])(out1)
        #     # word_re_embed,_ =  LSTM_CRF1(23922, 16, 768, 768, 1, 0.5, large_CRF=True, t = out1.shape[0]).to(DEVICE)(inputs[0],hidden=None)
        # print ('out1_',out1.shape)

        out1 = self.cov(out1)
        # out1 = self.apnb(out1)
        # out = out.permute(0,2,3,1)
        # print (out.shape)
        # b, c, h, w = out1.shape
        out1 = rearrange(out1, 'b c h w -> b c (h w)')
        out1 = out1.permute(0, 2, 1)
        # out1 = torch.cat((out1,pooling_output),dim=1)
        out1 = self.layernorm(out1)
        out1 = F.dropout(out1, p=0.5, training=self.training)
        # print(2)
        # print(out1.shape)
        output = self.relu(self.linear(out1))
        output = F.dropout(output, p=0.4, training=self.training)
        output = self.obj_output(output)  # [batch_size, maxlen, 2*plen]
        # print(3)
        # print(output.shape)
        ######
        # logits_output = torch.unsqueeze(logits, dim = 1)
        # final_output = logits_output + output
        output = torch.sigmoid(output)
        # output = output ** 2

        obj_preds = output.view(-1, output.shape[1], len(predicate2id), 2)
        return sub_preds, obj_preds


net = REModel().to(DEVICE)
print(DEVICE)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


class ValidDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        t = self.data[i]
        # word_input, center_word = [],[]
        # print (t['triple_list'])
        if len(t['text']) > 254:
            t['text'] = t['text'][:254]
        x = tokenizer.tokenize(t['text'])
        x = ["[CLS]"] + x + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(x)

        seg_ids = [0] * len(token_ids)
        assert len(token_ids) == len(t['text']) + 2

        token_ids = torch.LongTensor(self.sequence_padding(token_ids, maxlen=maxlen))
        seg_ids = torch.LongTensor(self.sequence_padding(seg_ids, maxlen=maxlen))

        # tri = t['triple_list']
        # print('tri',tri)
        '''
        return {'token_ids':token_ids,
                'seg_ids':seg_ids,
                'text':t['text'],
                'triple_list':t['triple_list']}
        '''
        # return token_ids, seg_ids, list(t['text']), list(t['triple_list'])
        return token_ids, seg_ids, t

    def __len__(self):
        data_len = len(self.data)
        return data_len

    def sequence_padding(self, x, maxlen, padding=0):
        output = np.concatenate([x, [padding] * (maxlen - len(x))]) if len(x) < maxlen else np.array(x[:maxlen])
        return output


valid_dataset = ValidDataset(valid_data)

valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch2, shuffle=False, drop_last=True)


def extract_spoes(data, model, device):
    '''
    """抽取三元组"""
    if len(text) > 254:
        text = text[:254]
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    assert len(token_ids) == len(text) + 2
    seg_ids = [0] * len(token_ids)
    '''
    # print (data[2])
    # print (data['text'])
    # token_ids = data['token_ids']
    token_ids = data[0]

    # seg_ids = data['seg_ids']
    seg_ids = data[1]
    # import pdb
    # pdb.set_trace()
    sub_preds = model(token_ids.to(device),
                      seg_ids.to(device))
    sub_preds = sub_preds.detach().cpu().numpy()  # [1, maxlen, 2]
    # print(sub_preds[0,])
    start = np.where(sub_preds[0, :, 0] > 0.5)[0]
    end = np.where(sub_preds[0, :, 1] > 0.5)[0]
    # print(start, end)
    tmp_print = []
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
            tmp_print.append(data[2][i - 1: j])

    if subjects:
        spoes = []
        # print (len(subjects)) #只有2
        token_ids = np.repeat(token_ids, len(subjects), 0)  # [len_subjects, seqlen]
        # print(token_ids.shape)
        seg_ids = np.repeat(seg_ids, len(subjects), 0)
        subjects = np.array(subjects)  # [len_subjects, 2]
        # 传入subject 抽取object和predicate
        _, object_preds = model(token_ids.to(device),
                                seg_ids.to(device),
                                torch.LongTensor(subjects).to(device))
        object_preds = object_preds.detach().cpu().numpy()
        #         print(object_preds.shape)
        for sub, obj_pred in zip(subjects, object_preds):
            # obj_pred [maxlen, 55, 2]
            start = np.where(obj_pred[:, :, 0] > 0.3)
            end = np.where(obj_pred[:, :, 1] > 0.3)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((sub[0] - 1, sub[1] - 1), predicate1, (_start - 1, _end - 1))
                        )
                        break
        # print (spoes)
        return [(data[2][s[0]:s[1] + 1], id2predicate[str(p)], data[2][o[0]:o[1] + 1]) for s, p, o in spoes]
    else:
        return []


def evaluate(valid_data, valid_load, model, device):
    """评估函数，计算f1、precision、recall
    """
    # F1 = []
    # P = []
    # Re = []
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open("./data/CMED/dev_pred.json", 'w', encoding='utf-8')
    pbar = tqdm()
    # for d in data:
    # with torch.no_grad:
    # print (type(valid_load))
    # return
    for idx, data in tqdm(enumerate(valid_load)):

        input = data[0], data[1], data[2]['text'][0]
        # print(input)
        # input = data[0], data[1], valid_data[idx]['text'], valid_data[idx]['triple_list']
        R = extract_spoes(input, model, device)
        # print ('R:',R)
        T = valid_data[idx]['triple_list']
        '''
        tri = data[3]
        #tri = tuple(tri)
        T = []
        for tris in tri:
            temp = tuple()
            for i in tris:
                temp += i
            T.append(temp)
        '''
        # print ('tri:',tri)
        # print ('tri:',temp_tri)
        R = set(R)
        # print ('R',R)
        T = set(T)
        # print('T', R)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        # F1.append(f1)
        # P.append(precision)
        # Re.append(recall)
        pbar.update()
        pbar.set_description(
            'F1: %.5f, \tPrecision: %.5f, \tRecall: %.5f' % (f1, precision, recall)
        )

        if f1 > 0.5:
            s = json.dumps({
                'text': valid_data[idx]['text'],
                'triple_list': list(T),
                'triple_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            }, ensure_ascii=False, indent=4)
            f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


'''        
def evaluate(data, model, device):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open("/home/jason/EXP/NLP/triple_test/data/CMED/dev_pred.json", 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:    
        R = extract_spoes(d['text'], model, device)

        T = d['triple_list']
        #print (T)
        R = set(R)
        #print ('R',R)
        T = set(T)

        #T = set()
        #for item in T1:
        #  for i in item:
        #    T.add(i)

        #print ('T',T)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )

        if f1 > 0.5: 

          s = json.dumps({
              'text': d['text'],
              'triple_list': list(T),
              'triple_list_pred': list(R),
              'new': list(R - T),
              'lack': list(T - R),
          }, ensure_ascii=False, indent=4)
          f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall
'''
import sys
import os
class Logger(object):
    def __init__(self,fileN="default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

# def FocalLoss(input, target ,gamma=2,weight=None,reduction='mean'):
#     # def __init__(self,gamma=2,weight=None,reduction='mean'):
#     #     super(FocalLoss, self).__init__()
#     #     self.gamma = gamma
#     #     self.weight = weight
#     #     self.reduction = reduction
#     # def forward(self, output, target):
#     out_target = torch.stack([input[i,t] for i.type(torch.bool),t.type(torch.bool) in enumerate(target)])
#     probs = torch.sigmoid(out_target)
#     focal_weight = torch.pow(1-probs,gamma=2)
#
#     ce_loss = F.cross_entropy(input,target,weight=None,reduction='none')
#     focal_loss = focal_weight*ce_loss
#
#     if reduction == 'mean':
#         focal_loss = (focal_loss/focal_weight.sum()).sum()
#     elif reduction == 'sum':
#         focal_loss = focal_loss.sum()
#
#     return focal_loss
# class FocalLoss(nn.Module):
#
#     def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.eps = eps
#         self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
#
#     def forward(self, input, target):
#         logp = self.ce(input, target)
#         print('logp',logp)
#         p = torch.exp(-logp)
#         loss = (1 - p) ** self.gamma * logp
#         return loss.mean()

# def Dice_loss(inputs,target,beta=1,smooth=1e-5):
#     n,c,h = inputs.size()
#     nt,ht,wt = target.size()
#     if n!= nt and h!=wt:
#         inputs = F.interpolate(inputs,size=(ht,wt),mode="bilinear",align_corners=True)
#     temp_imputs = torch.softmax(inputs.transpose(1,2).transpose(2,3).contiguous().view(n,-1,c),-1)
#     temp_target = target.view(n,-1,ct)
#
#     #......................
#     #ice loss
#     #......................
#     tp = torch.sum(temp_target[...,:-1]*temp_imputs,axis=[0,1])
#     fp = torch.sum(temp_imputs,axis=[0,1])-tp
#     fn = torch.sum(temp_target[...,:-1],axis=[0,1])-tp
#
#     score = ((1+beta**2)*tp+smooth)/((1+beta**2)*tp+beta**2*fn+fp+smooth)
#     dice_loss = 1-torch.mean(score)
#     return dice_loss


# def dice_coeff(pred, target):
#     smooth = 1.
#     num = pred.size(0)
#     m1 = pred.view(num, -1)  # Flatten
#     m2 = target.view(num, -1)  # Flatten
#     intersection = (m1 * m2).sum()
#
#     return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


# def train(model, train_loader, optimizer, epoches, device):
#     # model.train()
#     torch.backends.cudnn.enabled = False
#     for _ in range(epoches):
#         print('epoch: ', _ + 1)
#         start = time.time()
#         train_loss_sum = 0.0
#         for batch_idx, x in tqdm(enumerate(train_loader)):
#             # token_ids, seg_ids, sub_ids = x[0].to(device), x[1].to(device), x[2].to(device)
#             token_ids, seg_ids, sub_ids = x[0].to(device), x[1].to(device), x[2].to(device)
#             # tokens_words, masks_out, head = x[5].to(device), x[6].to(device), x[7].to(device)
#             # print (token_ids.shape)
#
#             mask = (token_ids > 0).float()
#             mask = mask.to(device)  # zero-mask
#             sub_labels, obj_labels = x[3].float().to(device), x[4].float().to(device)
#             sub_preds, obj_preds = model(token_ids, seg_ids, sub_ids)
#             # (batch_size, maxlen, 2),  (batch_size, maxlen, 55, 2)
#
#             #计算loss
#             smooth = 1
#             intersection = sub_labels * sub_preds
#             sub_dice_eff = (2 * intersection.sum(1) + smooth) / (sub_preds.sum(1) + sub_labels.sum(1) + smooth)
#             # print(sub_dice_eff)
#             smooth = 1
# #             intersection2 = obj_labels * obj_preds
# #             obj_dice_eff = (2 * intersection2.sum(1) + smooth) / (obj_preds.sum(1) + obj_labels.sum(1) + smooth)
# #             # print(obj_dice_eff)
# #             beta = 1
# #             smooth = 1e-5
# #             p = torch.sigmoid(sub_preds)
# #             tp = torch.sum(sub_labels[..., :-1] * p, axis=[0, 1])
# #             # print(tp)
# #             fp = torch.sum(p, axis=[0, 1]) - tp
# #             # print(fp)
# #             fn = torch.sum(sub_labels[..., :-1], axis=[0, 1]) - tp
# #             score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
# #             sub_dice_loss = 1-torch.mean(score)
# #             # print(sub_dice_loss)
# #             ce_loss_sub = F.binary_cross_entropy(sub_preds, sub_labels, reduction='none')  # [bs, ml, 2]
# #             p_t = p*sub_labels + (1-p)*(1-sub_labels)
# #             gamma = 2
# #             loss_sub= ce_loss_sub*((1-p_t)**gamma)
#
#             q = torch.sigmoid(obj_preds)
#             # print(q)
#             tp = torch.sum(obj_labels[..., :-1] * q, axis=[0, 1])
#             # print(tp)
#             fp = torch.sum(q, axis=[0, 1]) - tp
#             # print(fp)
#             fn = torch.sum(obj_labels[..., :-1], axis=[0, 1]) - tp
#             score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
#             obj_dice_loss = 1 - torch.mean(score)
#             # print(obj_dice_loss)

            # loss_sub = torch.mean(loss_sub, 2)  # (batch_size, maxlen)
            # loss_sub = torch.sum(loss_sub * mask) / torch.sum(mask)
            # # print('loss_sub:',loss_sub)
            # q = torch.sigmoid(obj_preds)
            # ce_loss_obj = F.binary_cross_entropy(obj_preds, obj_labels, reduction='none')  # [bs, ml, 55, 2]
            # q_t = q * obj_labels + (1 - q) * (1 - obj_labels)
            # gamma = 2
            # loss_obj = ce_loss_obj * ((1 - q_t) ** gamma)
            # loss_obj = torch.sum(torch.mean(loss_obj, 3), 2)  # (bs, maxlen)
            # loss_obj = torch.sum(loss_obj * mask) / torch.sum(mask)
            # loss = loss_sub + loss_obj
            # loss_sub = dice_coeff(sub_preds, sub_labels)
            # loss_obj = dice_coeff(obj_preds, obj_labels)
            # loss = loss_sub+ loss_obj
#             # 计算loss
#             loss_sub = F.binary_cross_entropy(sub_preds, sub_labels, reduction='none')  # [bs, ml, 2]
#             loss_sub = torch.mean(loss_sub, 2)  # (batch_size, maxlen)
#             loss_sub = torch.sum(loss_sub * mask) / torch.sum(mask)
#             loss_obj = F.binary_cross_entropy(obj_preds, obj_labels, reduction='none')  # [bs, ml, 55, 2]
#             loss_obj = torch.sum(torch.mean(loss_obj, 3), 2)  # (bs, maxlen)
#             loss_obj = torch.sum(loss_obj * mask) / torch.sum(mask)
#             loss = loss_sub + loss_obj
#             optimizer.zero_grad()
#
#             loss.backward()
#             optimizer.step()
#             train_loss_sum += loss.cpu().item()
#             if (batch_idx + 1) % 31 == 0:
#                 print('loss: ', train_loss_sum / (batch_idx + 1), 'time: ', time.time() - start)
#
#         torch.save(net.state_dict(), "./checkpoints/best_re.pth")
#
#         with torch.no_grad():
#             # model.eval()
#             # print (valid_data[:5])
#             val_f1, pre, rec = evaluate(valid_data, valid_loader, net, device)
#
#             print('F1_score: %.5f, Precision: %.5f, Recall: %.5f' % (val_f1, pre, rec))
#         # sys.stdout = Logger('./datalog.txt')
#             re = tuple((val_f1, pre, rec))
#             with open("./result_Dice_loss.json","a",encoding='utf-8') as f:
#                  json.dump(re,f,indent=4,ensure_ascii=True)
#             # print("f1, pre, rec: ", val_f1, pre, rec)
class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.class_entropy(inputs, targets, reduction='none',ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.sum()

def train(model, train_loader, optimizer, epoches, device):
    # model.train()
    torch.backends.cudnn.enabled = False
    list = []
    for _ in range(epoches):
        # f = open("./test.txt", 'w+', encoding='utf-8')
        print('epoch: ', _ + 1)
        start = time.time()
        train_loss_sum = 0.0
        for batch_idx, x in tqdm(enumerate(train_loader)):
            # token_ids, seg_ids, sub_ids = x[0].to(device), x[1].to(device), x[2].to(device)
            token_ids, seg_ids, sub_ids = x[0].to(device), x[1].to(device), x[2].to(device)
            # tokens_words, masks_out, head = x[5].to(device), x[6].to(device), x[7].to(device)
            # print (token_ids.shape)

            mask = (token_ids > 0).float()
            mask = mask.to(device)  # zero-mask
            sub_labels, obj_labels = x[3].float().to(device), x[4].float().to(device)
            sub_preds, obj_preds = model(token_ids, seg_ids, sub_ids)
            # (batch_size, maxlen, 2),  (batch_size, maxlen, 55, 2)

            # 计算loss
            smooth = 1
            intersection2 = obj_labels * obj_preds
            obj_dice_eff = (2 * intersection2.sum(1) + smooth) / (obj_preds.sum(1) + obj_labels.sum(1) + smooth)
            # print(obj_dice_eff)
            beta = 1
            smooth = 1e-5
            p = torch.sigmoid(sub_preds)
            tp = torch.sum(sub_labels[..., :-1] * p, axis=[0, 1])
            # print(tp)
            fp = torch.sum(p, axis=[0, 1]) - tp
            # print(fp)
            fn = torch.sum(sub_labels[..., :-1], axis=[0, 1]) - tp
            score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
            sub_dice_loss = 1-torch.mean(score)
            # print(sub_dice_loss)
            ce_loss_sub = F.binary_cross_entropy(sub_preds, sub_labels, reduction='none')  # [bs, ml, 2]
            p_t = p*sub_labels + (1-p)*(1-sub_labels)
            gamma = 2
            loss_sub= ce_loss_sub*((1-p_t)**gamma)

            q = torch.sigmoid(obj_preds)
            # print(q)
            tp = torch.sum(obj_labels[..., :-1] * q, axis=[0, 1])
            # print(tp)
            fp = torch.sum(q, axis=[0, 1]) - tp
            # print(fp)
            fn = torch.sum(obj_labels[..., :-1], axis=[0, 1]) - tp
            score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
            obj_dice_loss = 1 - torch.mean(score)
            # print(obj_dice_loss)
            loss_sub = torch.mean(loss_sub, 2)  # (batch_size, maxlen)
            loss_sub = torch.sum(loss_sub * mask) / torch.sum(mask)
            # print('loss_sub:',loss_sub)
            q = torch.sigmoid(obj_preds)
            ce_loss_obj = F.binary_cross_entropy(obj_preds, obj_labels, reduction='none')  # [bs, ml, 55, 2]
            q_t = q * obj_labels + (1 - q) * (1 - obj_labels)
            gamma = 2
            loss_obj = ce_loss_obj * ((1 - q_t) ** gamma)
            loss_obj = torch.sum(torch.mean(loss_obj, 3), 2)  # (bs, maxlen)
            loss_obj = torch.sum(loss_obj * mask) / torch.sum(mask)

            # jiaochashang
            # loss_sub = F.binary_cross_entropy(sub_preds, sub_labels, reduction='none')  # [bs, ml, 2]
            # loss_sub = torch.mean(loss_sub, 2)  # (batch_size, maxlen)
            # loss_sub = torch.sum(loss_sub * mask) / torch.sum(mask)
            # loss_obj = F.binary_cross_entropy(obj_preds, obj_labels, reduction='none')  # [bs, ml, 55, 2]
            # loss_obj = torch.sum(torch.mean(loss_obj, 3), 2)  # (bs, maxlen)
            # loss_obj = torch.sum(loss_obj * mask) / torch.sum(mask)
            loss = loss_sub + loss_obj
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            train_loss_sum += loss.cpu().item()

            if (batch_idx + 1) % 31 == 0:
                print('loss: ', train_loss_sum / (batch_idx + 1), 'time: ', time.time() - start)
        list.append(train_loss_sum / (batch_idx + 1))
        torch.save(net.state_dict(), "./checkpoints/best_re.pth")

        with torch.no_grad():
            # model.eval()
            # print (valid_data[:5])
            val_f1, pre, rec = evaluate(valid_data, valid_loader, net, device)
            print('F1_score: %.5f, Precision: %.5f, Recall: %.5f' % (val_f1, pre, rec))
            # print("f1, pre, rec: ", val_f1, pre, rec)
    print(list)

# LOGGER = set_logging(name='test', level=logging.INFO, verbose=True)


if __name__ == '__main__':
    # net.load_state_dict(torch.load("RE/data/bert_re.pth"))
    train(net, train_loader, optimizer, 600, DEVICE)