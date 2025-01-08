import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import nltk
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import csv
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import timm
import random
from torchinfo import summary
from glob import glob
from torchvision.transforms import ToTensor
import time
import json
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from nltk.util import ngrams
from collections import Counter
nltk.download('punkt')
tf = ToTensor()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
encoder_name='efficientnetv2_s'
model_layer=1280
params={'image_size':300,
        'lr':2e-4,
        'beta1':0.5,
        'beta2':0.999,
        'batch_size':4,
        'epochs':10000,
        'image_count':50,
        'data_path':'../../data/PatchGastricADC22/',
        'test_csv':'test_captions.csv',
        'vocab_path':'../../data/PatchGastricADC22/vocab.pkl',
        'embed_size':1024,
        'hidden_size':256,
        'num_layers':4,}

class CustomDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, data_path,image_count,image_size, csv, class_dataset, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = data_path+'/'
        self.image_count=image_count
        self.df = pd.read_csv(data_path+csv)
        self.class_dataset=class_dataset
        self.vocab = vocab
        self.transform = transform
        self.image_size=image_size
    def trans(self,image):
        if random.random() > 0.5:
            transform = transforms.RandomHorizontalFlip(1)
            image = transform(image)
            
        if random.random() > 0.5:
            transform = transforms.RandomVerticalFlip(1)
            image = transform(image)
            
        return image
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        df = self.df
        vocab = self.vocab
        img_id=df.loc[index]
        image_path = glob(self.root+'f_patches_captions/'+img_id['id']+'/*.jpg')
        caption=img_id['text']
        images=torch.zeros(self.image_count,3,self.image_size,self.image_size)
        image_index = torch.randint(low=0, high=len(
            image_path)-1, size=(self.image_count,))
        count = 0
        for ind in image_index:
            image = Image.open(image_path[ind]).convert('RGB')
            if self.transform is not None:
                image=self.transform(image).to(device)
            images[count] = image
            count += 1
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return images, target

    def __len__(self):
        return len(self.df)
    

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def idx2word(vocab, indices):
    sentence = []
    
    indices=indices.cpu().numpy()
    
    for index in indices:
        word = vocab.idx2word[index.item()]
        sentence.append(word)
    return sentence

def word2sentence(words_list):
    sentence=''
    for word in words_list:
        if word.isalnum():
            sentence+=' '+word
        else:
            sentence+=word
    return sentence

class FeatureExtractor(nn.Module):
    """Feature extoractor block"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        cnn1= timm.create_model(encoder_name)
        self.feature_ex = nn.Sequential(*list(cnn1.children())[:-1])

    def forward(self, inputs):
        features = self.feature_ex(inputs)
        
        return features
    
class AttentionMILModel(nn.Module):
    def __init__(self, num_classes, image_feature_dim,feature_extractor_scale1: FeatureExtractor):
        super(AttentionMILModel, self).__init__()
        self.num_classes = num_classes
        self.image_feature_dim = image_feature_dim

        # Remove the classification head of the CNN model
        self.feature_extractor = feature_extractor_scale1
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(image_feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classification layer
        self.classification_layer = nn.Linear(image_feature_dim, num_classes)

    def forward(self, inputs):
        batch_size, num_tiles, channels, height, width = inputs.size()
        
        # Flatten the inputs
        inputs = inputs.view(-1, channels, height, width)
        
        # Feature extraction using the pre-trained CNN
        features = self.feature_extractor(inputs)  # Shape: (batch_size * num_tiles, 2048, 1, 1)
        
        # Reshape features
        features = features.contiguous().view(batch_size, num_tiles, -1)  # Shape: (batch_size, num_tiles, 2048)
        
        # Attention mechanism
        attention_weights = self.attention(features)  # Shape: (batch_size, num_tiles, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize attention weights
        
        # Apply attention weights to features
        attended_features = torch.sum(features * attention_weights, dim=1)  # Shape: (batch_size, 2048)
        
        # Classification layer
        logits = self.classification_layer(attended_features)  # Shape: (batch_size, num_classes)
        
        return logits  


class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads, hidden_size, num_layers, max_seq_length=100):
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_size))
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.linear = nn.Linear(embed_size, vocab_size)
        
    def forward(self, features, captions, teacher_forcing_ratio=1.0):
        """
        features: (batch_size, embed_size)
        captions: (batch_size, max_seq_length)
        """
        batch_size = features.size(0)
        max_seq_length = captions.size(1)
        
        # Output 저장을 위한 텐서 초기화
        outputs = torch.zeros(batch_size, max_seq_length, self.vocab_size).to(features.device)
        
        # features를 memory로 사용
        memory = features.unsqueeze(0)  # (1, batch_size, embed_size)
        
        # 첫 번째 입력 토큰은 <start> 토큰
        input_caption = captions[:, 0].unsqueeze(1)  # (batch_size, 1)
        
        for t in range(1, max_seq_length):
            # 임베딩 및 포지셔널 인코딩 적용
            input_embedded = self.embed(input_caption) + self.positional_encoding[:, :input_caption.size(1), :]
            input_embedded = input_embedded.permute(1, 0, 2)  # (seq_len, batch_size, embed_size)
            
            # 타겟 마스크 생성
            tgt_mask = self.generate_square_subsequent_mask(input_embedded.size(0)).to(features.device)
            
            # Transformer 디코더에 입력
            transformer_output = self.transformer_decoder(input_embedded, memory, tgt_mask=tgt_mask)
            transformer_output = transformer_output.permute(1, 0, 2)
            
            # 현재 시간 스텝의 출력 계산
            output = self.linear(transformer_output[:, -1, :])  # (batch_size, vocab_size)
            outputs[:, t, :] = output  # 출력 저장
            
            # 다음 입력 결정 (교사 강요 비율에 따라)
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                # 실제 캡션의 다음 토큰 사용
                next_input = captions[:, t].unsqueeze(1)
            else:
                # 모델의 예측 사용
                _, predicted = output.max(1)
                next_input = predicted.unsqueeze(1)
            
            # 다음 입력을 input_caption에 추가
            input_caption = torch.cat([input_caption, next_input], dim=1)
        
        return outputs

    def generate_square_subsequent_mask(self, sz):
        """시퀀스의 순차적인 마스크 생성"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def sample(self, features, max_seq_length=None):
        """Greedy Search 방식으로 시퀀스를 샘플링합니다."""
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        
        batch_size = features.size(0)
        sampled_ids = []
        
        # 첫 번째 토큰은 <start> 토큰
        input_caption = torch.ones(batch_size, 1).long().to(features.device)
        memory = features.unsqueeze(0)  # (1, batch_size, embed_size)
        
        for _ in range(max_seq_length):
            input_embedded = self.embed(input_caption) + self.positional_encoding[:, :input_caption.size(1), :]
            input_embedded = input_embedded.permute(1, 0, 2)
            tgt_mask = self.generate_square_subsequent_mask(input_embedded.size(0)).to(features.device)
            transformer_output = self.transformer_decoder(input_embedded, memory, tgt_mask=tgt_mask)
            transformer_output = transformer_output.permute(1, 0, 2)
            output = self.linear(transformer_output[:, -1, :])  # (batch_size, vocab_size)
            _, predicted = output.max(1)
            sampled_ids.append(predicted)
            input_caption = torch.cat([input_caption, predicted.unsqueeze(1)], dim=1)
        
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids
    
def bleu_n(pred_words_list,label_words_list):

    bleu1 = sentence_bleu([label_words_list], pred_words_list, weights=(1, 0, 0, 0))


# BLEU@2 calculation
    bleu2 = sentence_bleu([label_words_list], pred_words_list, weights=(0.5, 0.5, 0.0, 0.0))


    bleu3=sentence_bleu([label_words_list], pred_words_list, weights=(0.33, 0.33, 0.33, 0.0))


    bleu4=sentence_bleu([label_words_list], pred_words_list, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1,bleu2,bleu3,bleu4
def rouge_scores(pred_sentence, label_sentence):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate the scores
    scores = scorer.score(label_sentence, pred_sentence)
    
    # Extract the precision, recall, and f1 scores for each ROUGE metric
    rouge1 = scores['rouge1']
    rouge2 = scores['rouge2']
    rougeL = scores['rougeL']
    
    # Return the ROUGE scores
    return rouge1, rouge2, rougeL

def calculate_cider(references, hypotheses):
    """
    CIDEr 점수를 계산합니다.

    Args:
        references (list of str): 각 이미지의 참조 설명 문장들.
        hypotheses (list of str): 각 이미지에 대한 생성된 설명 문장들.

    Returns:
        float: CIDEr 점수의 평균값.
        list of float: 각 이미지에 대한 CIDEr 점수.
    """
    # pycocoevalcap 라이브러리는 references와 hypotheses의 형식을 다음과 같이 요구합니다.
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [hypothesis] for i, hypothesis in enumerate(hypotheses)}

    # CIDEr 계산
    cider_scorer = Cider()
    cider_score, cider_per_image_scores = cider_scorer.compute_score(gts, res)

    return cider_score, cider_per_image_scores

# 1/2-gram Precision 계산 함수
def calculate_ngram_precision(references, hypotheses, n):
    """
    Calculates n-gram precision for a set of generated captions against reference captions.

    Args:
        references (list of str): List of reference captions.
        hypotheses (list of str): List of generated captions.
        n (int): The n-gram size.

    Returns:
        list of float: Precision scores for each hypothesis.
    """
    precision_scores = []

    for hypothesis, reference in zip(hypotheses, references):
        # Tokenize hypothesis and reference
        hypothesis_tokens = hypothesis.split()
        reference_tokens = reference.split()

        # Generate n-grams
        hypothesis_ngrams = list(ngrams(hypothesis_tokens, n))
        reference_ngrams = list(ngrams(reference_tokens, n))

        # Count n-gram occurrences
        hypothesis_counts = Counter(hypothesis_ngrams)
        reference_counts = Counter(reference_ngrams)

        # Calculate overlapping n-grams
        overlap = sum((hypothesis_counts & reference_counts).values())
        total = sum(hypothesis_counts.values())

        # Precision calculation
        precision = overlap / total if total > 0 else 0.0
        precision_scores.append(precision)

    return precision_scores

# Mean and SD Calculation for Average 1/2-gram Precision
def calculate_mean_sd_average_precision(references, hypotheses):
    """
    Calculates mean and standard deviation for the average of 1-gram and 2-gram precision.

    Args:
        references (list of str): List of reference captions.
        hypotheses (list of str): List of generated captions.

    Returns:
        dict: Mean and SD for the average 1/2-gram precision.
    """
    avg_precisions = []

    for hypothesis, reference in zip(hypotheses, references):
        precisions = []
        for n in [1, 2]:
            ngram_precision = calculate_ngram_precision([reference], [hypothesis], n)
            precisions.append(ngram_precision[0])
        avg_precisions.append(np.mean(precisions))

    mean_avg_precision = np.mean(avg_precisions)
    sd_avg_precision = np.std(avg_precisions)

    return {"mean": mean_avg_precision, "sd": sd_avg_precision}
with open(params['vocab_path'], 'rb') as f:
        vocab = pickle.load(f)
transform = transforms.Compose([ 
        transforms.RandomCrop((params['image_size'],params['image_size'])),
        transforms.ToTensor()])


test_dataset=CustomDataset(params['data_path'],params['image_count'],params['image_size'],params['test_csv'],'test',vocab,transform=transform)

test_dataloader=DataLoader(test_dataset,batch_size=params['batch_size'],shuffle=False,collate_fn=collate_fn)
Feature_Extractor=FeatureExtractor()
encoder = AttentionMILModel(params['embed_size'],model_layer,Feature_Extractor).to(device)
decoder = DecoderTransformer(params['embed_size'], len(vocab), 16, params['hidden_size'], params['num_layers']).to(device).to(device)
criterion = nn.CrossEntropyLoss()
model_param = list(decoder.parameters()) + list(encoder.parameters())
optimizer = torch.optim.Adam(model_param, lr=params['lr'], betas=(params['beta1'], params['beta2']))
encoder.load_state_dict(torch.load('../../model/'+encoder_name+'_and_transformer_'+str(params['image_count'])+'_encoder_check.pth',map_location=device))
decoder.load_state_dict(torch.load('../../model/'+encoder_name+'_and_transformer_'+str(params['image_count'])+'_decoder_check.pth',map_location=device))
encoder.eval()
decoder.eval()
std_dict={'bleu4':[],'meteor':[],'rougeL':[],'cider':[]}
mean_dict={'bleu4':0,'meteor':0,'rougeL':0,'cider':0}
roop=30
with torch.no_grad():
    for i in range(roop):
        total_bleu=[]
        total_meteor=[]

        total_Rogue=[]
        total_reference=[]
        total_candidate=[]
        val_count = 0
        val_loss = 0.0 
        val_bleu_score = 0.0
        val = tqdm(test_dataloader)
        for images, captions, lengths in val:
            
            images = images.to(device)
            captions = captions.to(device)
            
            # Encoder를 통해 특징 추출
            features = encoder(images)
            
            # 디코더에 입력 (손실 계산을 위해 교사 강요 적용)
            outputs = decoder(features, captions, teacher_forcing_ratio=0.5)
            
            # 손실 계산을 위한 정렬
            captions_target = captions[:, 1:]
            outputs = outputs[:, 1:, :]
            outputs = outputs.reshape(-1, outputs.size(2))
            targets = captions_target.reshape(-1)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            # 캡션 생성 (교사 강요 없이)
            sampled_ids = decoder.sample(features)
            
            # BLEU 점수 계산
            val_count += 1
            for i in range(images.size(0)):
                
                predicted_caption = idx2word(vocab, sampled_ids[i])
                target_caption = idx2word(vocab, captions[i])
                
                # 특수 토큰 제거
                predicted_caption = [word for word in predicted_caption if word not in ['<start>', '<end>', '<pad>','<unk>']]
                target_caption = [word for word in target_caption if word not in ['<start>', '<end>', '<pad>','<unk>']]
                reference = word2sentence(target_caption )
                candidate = word2sentence(predicted_caption)
                total_reference.append(reference)
                total_candidate.append(candidate)
                bleu1,bleu2,bleu3,bleu4=bleu_n(predicted_caption,target_caption)
                rouge1, rouge2, rougeL = rouge_scores(candidate, reference)
                total_bleu.append([bleu1,bleu2,bleu3,bleu4])
                total_Rogue.append([rouge1, rouge2, rougeL])
                meteor_score_value = meteor_score([target_caption], predicted_caption)
                total_meteor.append(meteor_score_value)
                
                # BLEU-4 점수 계산
                bleu_score = sentence_bleu([target_caption], predicted_caption, weights=(0.25, 0.25, 0.25, 0.25))
                val_bleu_score += bleu_score
            val.set_description(f"Step: {val_count} loss : {val_loss/val_count:.4f} BLEU-4: {val_bleu_score/(val_count*params['batch_size']):.4f}")
        average_cider_score, cider_scores = calculate_cider(total_reference, total_candidate)
        average_result = calculate_mean_sd_average_precision(total_reference, total_candidate)
        Avg_gram_mean=average_result['mean']
        Avg_gram_sd=average_result['sd']
        mean_dict['bleu4']+=val_bleu_score/(val_count*params['batch_size'])
        mean_dict['meteor']+=np.mean(total_meteor)
        mean_dict['rougeL']+=np.array(total_Rogue)[:,2].mean()
        mean_dict['cider']+=average_cider_score
        
        std_dict['bleu4'].append(val_bleu_score/(val_count*params['batch_size']))
        std_dict['meteor'].append(np.mean(total_meteor))
        std_dict['rougeL'].append(np.array(total_Rogue)[:,2].mean())
        std_dict['cider'].append(average_cider_score)
    mean_dict['bleu4']/=roop
    mean_dict['meteor']/=roop
    mean_dict['rougeL']/=roop
    mean_dict['cider']/=roop

print(f'Bleu-4:{np.array(std_dict["bleu4"]).min():.3f}+-{np.array(std_dict["bleu4"]).std():.3f} \nMeteor:{np.array(std_dict["meteor"]).min():.3f}+-{np.array(std_dict["meteor"]).std():.3f} \nRougeL:{np.array(std_dict["rougeL"]).min():.3f}+-{np.array(std_dict["rougeL"]).std():.3f} \nCider:{np.array(std_dict["cider"]).min():.3f}+-{np.array(std_dict["cider"]).std():.3f}')