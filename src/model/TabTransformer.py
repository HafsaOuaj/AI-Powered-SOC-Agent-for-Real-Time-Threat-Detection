import torch.nn as nn
import torch
import torch.nn.functional as F


class TabTransformer(nn.Module):


    def __init__(self,categorical_cardinalities,num_numerical, emb_dim=16, num_heads=2, ff_hidden=64):
        super(TabTransformer,self).__init__()

        # Embedding Layers

        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality,emb_dim) for cardinality in categorical_cardinalities
        ])

        self.num_categoricals =  len(categorical_cardinalities)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,nhead=num_heads,batch_first=True)
        self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=1)
        
        # LayerNorm for numerical features
        self.norm_num = nn.LayerNorm(num_numerical)

        # MLP classifier
        self.fc = nn.Sequential(
            nn.Linear(emb_dim*self.num_categoricals+num_numerical,ff_hidden),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.Linear(ff_hidden,1),
            nn.Sigmoid()
        )

    
    def forward(self,x_cat,x_num):
        embeds= [emb(x_cat[:,i])for i,emb in enumerate(self.embeddings)]
        x_cat_emb=torch.stack(embeds,dim=1)

        x_transformed=self.transformer(x_cat_emb)
        x_flat=x_transformed.flatten(1)
        x_num_norm=self.norm_num(x_num)
        x_full = torch.cat([x_flat,x_num_norm],dim=1)

        return self.fc(x_full)

