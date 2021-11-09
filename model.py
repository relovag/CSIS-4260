import timm
import torch.nn as nn
import transformers


class ImageEncoder(nn.Module):
    def __init__(self, img_enc, embed_size):
        super(ImageEncoder, self).__init__()
        self.img_enc = timm.create_model(img_enc, pretrained=True)
        self.n_feats = self.img_enc.classifier.in_features
        self.img_enc.reset_classifier(0)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.n_feats, embed_size)
        self.freeze()

    def forward(self, imgs):
        img_feats = self.img_enc(imgs)
        return self.fc(self.dropout(img_feats))

    def freeze(self):
        for params in self.img_enc.parameters():
            params.requires_grad = False
        self.fc.weight.requires_grad = True
        self.fc.bias.requires_grad = True


class TextEncoder(nn.Module):
    def __init__(self, text_enc, embed_size):
        self.text_enc = transformers.AutoModel.from_pretrained(text_enc)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, embed_size)

    def forward(self, ids, mask):
        txt_feats = self.text_enc(input_ids=ids, attention_mask=mask,
                                  output_hidden_states=False)
        return self.fc(self.dropout(txt_feats[1]))

    def freeze(self):
        for params in self.text_enc.parameters():
            params.requires_grad = False
        self.fc.weight.requires_grad = True
        self.fc.bias.requires_grad = True
