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
from torch.utils.data import DataLoader, Dataset
import csv
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
import timm
from torchinfo import summary
from glob import glob

nltk.download('punkt')
# LD_LIBRARY_PATH 환경 변수 해제
if 'LD_LIBRARY_PATH' in os.environ:
    del os.environ['LD_LIBRARY_PATH']
# Device configurationresul
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


params = {'image_size': 256,
          'lr': 2e-4,
          'beta1': 0.5,
          'beta2': 0.999,
          'batch_size': 8,
          'epochs': 10000,
          'image_count': 25,
          'data_path': '../../data/PatchGastricADC22/',
          'train_csv': 'train_captions.csv',
          'val_csv': 'test_captions.csv',
          'vocab_path': '../../data/PatchGastricADC22/vocab.pkl',
          'embed_size': 300,
          'hidden_size': 256,
          'num_layers': 1, }


class CustomDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, data_path, image_count, image_size, csv, class_dataset, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = data_path+'/'
        self.image_count = image_count
        self.df = pd.read_csv(data_path+csv)
        self.class_dataset = class_dataset
        self.vocab = vocab
        self.transform = transform
        self.image_size = image_size

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        df = self.df
        vocab = self.vocab
        img_id = df.loc[index]
        image_path = glob(self.root+'patches_captions/'+img_id['id']+'*.jpg')
        caption = img_id['text']
        images = torch.zeros(self.image_count, 3,
                             self.image_size, self.image_size)
        image_index = torch.randint(low=0, high=len(
            image_path)-1, size=(self.image_count,))
        count = 0
        for index in image_index:
            image = Image.open(image_path[index]).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
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
    for i in range(params['batch_size']):
        indices[i].cpu().numpy()

    for index in indices:
        word = vocab.idx2word[index]
        sentence.append(word)
    return sentence


class FeatureExtractor(nn.Module):
    """Feature extoractor block"""

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        cnn1 = timm.create_model(
            'vit_base_patch16_siglip_256', pretrained=True)
        self.feature_ex = nn.Sequential(*list(cnn1.children())[:-1])

    def forward(self, inputs):
        features = self.feature_ex(inputs)

        return features


class AttentionMILModel(nn.Module):
    def __init__(self, num_classes, image_feature_dim, feature_extractor_scale1: FeatureExtractor):
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
        # Shape: (batch_size * num_tiles, 2048, 1, 1)
        features = self.feature_extractor(inputs)

        # Reshape features
        # Shape: (batch_size, num_tiles, 2048)
        features = features.view(batch_size, num_tiles, -1)

        # Attention mechanism
        # Shape: (batch_size, num_tiles, 1)
        attention_weights = self.attention(features)
        # Normalize attention weights
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention weights to features
        # Shape: (batch_size, 2048)
        attended_features = torch.sum(features * attention_weights, dim=1)

        # Classification layer
        # Shape: (batch_size, num_classes)
        logits = self.classification_layer(attended_features)

        return logits


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True)  # change for LSTM or RNN
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            # hiddens: (batch_size, 1, hidden_size)
            hiddens, states = self.lstm(inputs, states)
            # outputs:  (batch_size, vocab_size)
            outputs = self.linear(hiddens.squeeze(1))
            # predicted: (batch_size)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)

            # inputs: (batch_size, embed_size)
            inputs = self.embed(predicted)
            # inputs: (batch_size, 1, embed_size)
            inputs = inputs.unsqueeze(1)
        # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

    def stochastic_sample(self, features, temperature, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            # hiddens: (batch_size, 1, hidden_size)
            hiddens, states = self.lstm(inputs, states)
            # outputs:  (batch_size, vocab_size)
            outputs = self.linear(hiddens.squeeze(1))

            soft_out = F.softmax(outputs/temperature, dim=1)
            predicted = torch.multinomial(soft_out, 1).view(1)

            sampled_ids.append(predicted)

            # inputs: (batch_size, embed_size)
            inputs = self.embed(predicted)
            # inputs: (batch_size, 1, embed_size)
            inputs = inputs.unsqueeze(1)
        # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


with open(params['vocab_path'], 'rb') as f:
    vocab = pickle.load(f)
transform = transforms.Compose([
    transforms.RandomCrop(params['image_size']),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

train_dataset = CustomDataset(params['data_path'], params['image_count'],
                              params['image_size'], params['train_csv'], 'train', vocab, transform=transform)
test_dataset = CustomDataset(params['data_path'], params['image_count'],
                             params['image_size'], params['val_csv'], 'val', vocab, transform=transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(
    test_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn)

Feature_Extractor = FeatureExtractor()
encoder = AttentionMILModel(300, 768, Feature_Extractor).to(device)
decoder = DecoderRNN(params['embed_size'], params['hidden_size'], len(
    vocab), params['num_layers']).to(device)
criterion = nn.CrossEntropyLoss()
model_param = list(decoder.parameters()) + list(encoder.parameters())
optimizer = torch.optim.Adam(
    model_param, lr=params['lr'], betas=(params['beta1'], params['beta2']))

plt_count = 0
sum_loss = 1000.0
for epoch in range(params['epochs']):
    train = tqdm(train_dataloader)
    count = 0
    train_loss = 0.0
    for images, captions, lengths in train:
        count += 1
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        outputs = outputs
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train.set_description(
            f"epoch: {epoch+1}/{params['epochs']} Step: {count+1} loss : {train_loss/count:.4f} ")
    with torch.no_grad():
        val_count = 0
        val_loss = 0.0
        val_bleu_loss = 0.0
        val = tqdm(val_dataloader)
        for images, captions, lengths in val:
            val_count += 1
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(
                captions, lengths, batch_first=True)[0]
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            outputs = outputs
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val.set_description(
                f"epoch: {epoch+1}/{params['epochs']} Step: {val_count+1} loss : {val_loss/val_count:.4f} ")
    if val_loss < sum_loss:
        sum_loss = val_loss
        torch.save(encoder.state_dict(),
                   '../../model/MILattention_lstm/vit_encoder_check.pth')
        torch.save(decoder.state_dict(),
                   '../../model/MILattention_lstm/vit_decoder_check.pth')
