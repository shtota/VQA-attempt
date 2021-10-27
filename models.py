import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F

WORD_EMBED_SIZE = 300
NUM_LSTM_LAYERS = 2
HIDDEN_LSTM_SIZE = 512
IMAGE_FEATURES = 2048
MID_FEATURES = 1024
A_MID_FEATURES = 512
DROPOUT = 0.5
GLIMPSES = 2
TEXT_FEATURES = NUM_LSTM_LAYERS * HIDDEN_LSTM_SIZE


class ImagePreprocesser(nn.Module):
    def __init__(self):
        super(ImagePreprocesser, self).__init__()
        self.model = models.resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output

        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer

class TextEncoder(nn.Module):
    def __init__(self, input_text_size):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(input_text_size, WORD_EMBED_SIZE)
        self.activation = nn.Tanh()
        self.drop = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(WORD_EMBED_SIZE, HIDDEN_LSTM_SIZE, NUM_LSTM_LAYERS)
        nn.init.xavier_uniform_(self.embedding.weight)
        for w in self.lstm.weight_ih_l0.chunk(4, 0):
            nn.init.xavier_uniform_(w)
        for w in self.lstm.weight_hh_l0.chunk(4, 0):
            nn.init.xavier_uniform_(w)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

    def forward(self, question):
        qst_vec = self.embedding(question)
        qst_vec = self.activation(self.drop(qst_vec))
        qst_vec = qst_vec.transpose(0, 1)
        _, (hidden, cell) = self.lstm(qst_vec)
        qst_feature = cell.transpose(0, 1)
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)

        return qst_feature


class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.image_convolution = nn.Conv2d(IMAGE_FEATURES, A_MID_FEATURES, 1, bias=False)  # let self.lin take care of bias
        self.question_linear = nn.Linear(TEXT_FEATURES, A_MID_FEATURES)
        self.result_convolution = nn.Conv2d(A_MID_FEATURES, GLIMPSES, 1)

        self.drop = nn.Dropout(DROPOUT)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img, qst):
        img = self.image_convolution(self.drop(img))
        qst = self.question_linear(self.drop(qst))
        attention_question = tile_2d_over_nd(qst, img)
        result = self.relu(img + attention_question)
        result = self.result_convolution(self.drop(result))
        return result


class VqaModel(nn.Module):
    def __init__(self, input_text_size, output_text_size):
        super(VqaModel, self).__init__()
        print('Using encoded images')
        self.question_encoder = TextEncoder(input_text_size)
        self.activation = nn.Tanh()
        self.attention = AttentionModel()
        self.dropout1 = nn.Dropout(DROPOUT)
        self.dropout2 = nn.Dropout(DROPOUT)

        self.linear1 = nn.Linear(GLIMPSES * IMAGE_FEATURES + TEXT_FEATURES, TEXT_FEATURES)
        self.activation2 = nn.ReLU()
        self.linear2 = nn.Linear(TEXT_FEATURES, output_text_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, img, qst):
        l2_norm = img.norm(p=2, dim=1, keepdim=True).detach()
        img_ = img.div(l2_norm)

        qst_ = self.question_encoder(qst)
        attention = self.attention(img_, qst_)
        img_ = apply_attention(img_, attention)
        combined = torch.cat([img_, qst_], dim=1)
        combined = self.dropout1(combined)
        combined = self.linear1(combined)
        combined = self.activation2(combined)

        combined = self.dropout2(combined)
        combined = self.linear2(combined)
        return combined

def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled

def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
    weighted = attention * input # [n, g, v, s]
    weighted_mean = weighted.sum(dim=-1) # [n, g, v]
    return weighted_mean.view(n, -1)