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
from nltk.translate.bleu_score import sentence_bleu
import timm
import random
from torchinfo import summary
from glob import glob
from torchvision.transforms import ToTensor
import time
import json
nltk.download('punkt')
tf = ToTensor()
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
encoder_name='efficientnetv2_s'
model_layer=1280
params={'image_size':300,
        'lr':2e-4,
        'beta1':0.5,
        'beta2':0.999,
        'batch_size':4,
        'epochs':10000,
        'image_count':25,
        'data_path':'../../data/PatchGastricADC22/',
        'train_csv':'train_captions.csv',
        'val_csv':'test_captions.csv',
        'vocab_path':'../../data/PatchGastricADC22/vocab.pkl',
        'embed_size':512,
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
                image = self.trans(image)
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
    
    aa=indices.cpu().numpy()
    
    for index in aa:
        word = vocab.idx2word[index]
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

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers, max_seq_length=100):
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Positional Encoding (optional)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_size))

    def forward(self, features, captions, teacher_forcing_ratio=1.0):
        """
        features: (batch_size, embed_size)
        captions: (batch_size, max_seq_length)
        """
        batch_size = features.size(0)
        max_seq_length = captions.size(1)
        
        # LSTM의 초기 hidden state와 cell state
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(features.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(features.device)
        
        # 첫 번째 입력은 <start> 토큰
        input_caption = captions[:, 0].unsqueeze(1)  # (batch_size, 1)
        
        # Output 저장
        outputs = torch.zeros(batch_size, max_seq_length, self.vocab_size).to(features.device)
        
        for t in range(1, max_seq_length):
            # Embedding과 Positional Encoding
            input_embedded = self.embed(input_caption)  # (batch_size, 1, embed_size)
            if self.positional_encoding is not None:
                input_embedded += self.positional_encoding[:, :1, :]
            
            # LSTM 한 스텝 실행
            lstm_out, (h, c) = self.lstm(input_embedded, (h, c))  # (batch_size, 1, hidden_size)
            
            # 현재 시간 스텝의 출력 계산
            output = self.linear(lstm_out.squeeze(1))  # (batch_size, vocab_size)
            outputs[:, t, :] = output
            
            # Teacher Forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                # 다음 입력으로 정답 캡션 사용
                next_input = captions[:, t].unsqueeze(1)
            else:
                # 다음 입력으로 모델의 예측 사용
                _, predicted = output.max(1)
                next_input = predicted.unsqueeze(1)
            
            input_caption = next_input
        
        return outputs

    def sample(self, features, max_seq_length=None):
        """
        Greedy Search를 사용한 시퀀스 샘플링
        """
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        
        batch_size = features.size(0)
        sampled_ids = []
        
        # LSTM 초기 상태
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(features.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(features.device)
        
        # 첫 입력은 <start> 토큰
        input_caption = torch.ones(batch_size, 1).long().to(features.device)  # <start> token ID
        
        for _ in range(max_seq_length):
            # Embedding과 Positional Encoding
            input_embedded = self.embed(input_caption)  # (batch_size, 1, embed_size)
            if self.positional_encoding is not None:
                input_embedded += self.positional_encoding[:, :1, :]
            
            # LSTM 한 스텝 실행
            lstm_out, (h, c) = self.lstm(input_embedded, (h, c))
            
            # 출력 계산
            output = self.linear(lstm_out.squeeze(1))  # (batch_size, vocab_size)
            _, predicted = output.max(1)  # 가장 확률이 높은 단어 선택
            
            sampled_ids.append(predicted)
            input_caption = predicted.unsqueeze(1)  # 다음 입력으로 업데이트
        
        sampled_ids = torch.stack(sampled_ids, 1)  # (batch_size, max_seq_length)
        return sampled_ids

    
def bleu_n(pred_words_list,label_words_list):

    bleu1 = sentence_bleu([label_words_list], pred_words_list, weights=(1, 0, 0, 0))


# BLEU@2 calculation
    bleu2 = sentence_bleu([label_words_list], pred_words_list, weights=(0, 1, 0, 0))


    bleu3=sentence_bleu([label_words_list], pred_words_list, weights=(0, 0, 1, 0))


    bleu4=sentence_bleu([label_words_list], pred_words_list, weights=(0, 0, 0, 1))
    return bleu1,bleu2,bleu3,bleu4

with open(params['vocab_path'], 'rb') as f:
        vocab = pickle.load(f)
transform = transforms.Compose([ 
        transforms.Resize((params['image_size'],params['image_size'])),
        transforms.ToTensor()])

train_dataset=CustomDataset(params['data_path'],params['image_count'],params['image_size'],params['train_csv'],'train',vocab,transform=transform)
test_dataset=CustomDataset(params['data_path'],params['image_count'],params['image_size'],params['val_csv'],'val',vocab,transform=transform)
train_dataloader=DataLoader(train_dataset,batch_size=params['batch_size'],shuffle=True,collate_fn=collate_fn)
val_dataloader=DataLoader(test_dataset,batch_size=params['batch_size'],shuffle=True,collate_fn=collate_fn)


Feature_Extractor=FeatureExtractor()
encoder = AttentionMILModel(params['embed_size'],model_layer,Feature_Extractor).to(device)
decoder =  DecoderLSTM(params['embed_size'], len(vocab), 16, params['hidden_size'], params['num_layers']).to(device).to(device)
criterion = nn.CrossEntropyLoss()
model_param = list(decoder.parameters()) + list(encoder.parameters())
optimizer = torch.optim.AdamW(model_param, lr=params['lr'], betas=(params['beta1'], params['beta2']))
# summary(encoder, input_size=(params['batch_size'],50, 3, params['image_size'], params['image_size']))


plt_count=0
sum_loss= 0
scheduler = 0.90
teacher_forcing=0.0
import random  # random 모듈 임포트

for epoch in range(params['epochs']):
    train = tqdm(train_dataloader)
    count = 0
    train_loss = 0.0
    
    # 에폭마다 teacher_forcing_ratio 조정 (예: 점진적으로 감소)
    teacher_forcing_ratio = 0.95 ** epoch  # 지수적 감소
    teacher_forcing_ratio = max(0.2, teacher_forcing_ratio)
    encoder.train()
    decoder.train()
    for images, captions, lengths in train:
        count += 1
        images = images.to(device)
        captions = captions.to(device)
        
        # Encoder를 통해 특징 추출
        features = encoder(images)
        
        # 디코더에 입력 (teacher_forcing_ratio 적용)
        outputs = decoder(features, captions, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # 출력 및 타겟의 차원 맞추기
        captions_target = captions[:, 1:]  # 첫 번째 토큰(<start>) 제외
        outputs = outputs[:, 1:, :]  # 첫 번째 출력 제외
        outputs = outputs.reshape(-1, outputs.size(2))  # (batch_size * seq_length, vocab_size)
        targets = captions_target.reshape(-1)  # (batch_size * seq_length)
        
        # 손실 계산
        loss = criterion(outputs, targets)
        
        # 역전파 및 옵티마이저 스텝
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train.set_description(f"train epoch: {epoch+1}/{params['epochs']} Step: {count} loss : {train_loss/count:.4f}")
    encoder.eval()
    decoder.eval()  
    with torch.no_grad():
        val_count = 0
        val_loss = 0.0 
        val_bleu_score = 0.0
        val = tqdm(val_dataloader)
        for images, captions, lengths in val:
            
            images = images.to(device)
            captions = captions.to(device)
            
            # Encoder를 통해 특징 추출
            features = encoder(images)
            
            # 디코더에 입력 (손실 계산을 위해 교사 강요 적용)
            outputs = decoder(features, captions, teacher_forcing_ratio=teacher_forcing_ratio)
            
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
            for i in range(images.size(0)):
                
                predicted_caption = idx2word(vocab, sampled_ids[i])
                target_caption = idx2word(vocab, captions[i])
                
                # 특수 토큰 제거
                predicted_caption = [word for word in predicted_caption if word not in ['<start>', '<end>', '<pad>']]
                target_caption = [word for word in target_caption if word not in ['<start>', '<end>', '<pad>']]
                
                # BLEU-4 점수 계산
                bleu_score = sentence_bleu([target_caption], predicted_caption, weights=(0.25, 0.25, 0.25, 0.25))
                val_bleu_score += bleu_score
            val_count += 1
            val.set_description(f"val epoch: {epoch+1}/{params['epochs']} Step: {val_count} loss : {val_loss/val_count:.4f} BLEU-1: {val_bleu_score/(val_count):.4f}")
    if val_bleu_score/val_count>sum_loss:
        sum_loss=val_bleu_score/val_count
        torch.save(encoder.state_dict(), '../../model/'+encoder_name+'_and_lstm_encoder_check.pth')
        torch.save(decoder.state_dict(), '../../model/'+encoder_name+'_and_lstm_decoder_check.pth')