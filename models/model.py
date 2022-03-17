import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

        self.ip = 0.01
        self.dr = args.dropout

        self.dropout2 = nn.Dropout(p=self.dr)  # dropout ratio

        # _______ User embedding + Item embedding
        self.u_emb = nn.Embedding(args.user_size, args.embed_size, sparse=False)
        self.i_emb = nn.Embedding(args.item_size, args.embed_size, sparse=False)

        # _______ Feature embedding and Preference embedding are common_______
        self.cate_emb = nn.Embedding(args.cate_pad + 1, args.embed_size, padding_idx=-1) 
        self.attr_emb = nn.Embedding(args.attr_pad + 1, args.embed_size, padding_idx=-1) 

        # _______ Feature embedding and Preference embedding are common_______
        self.cate_emb_context = nn.Embedding(args.cate_pad + 1, args.embed_size, padding_idx=-1) 
        self.attr_emb_context = nn.Embedding(args.attr_pad + 1, args.embed_size, padding_idx=-1) 

        # _______ Feature embedding and Preference embedding are common_______
        self.cate_emb_target = nn.Embedding(args.cate_pad + 1, args.embed_size, padding_idx=-1) 
        self.attr_emb_target = nn.Embedding(args.attr_pad + 1, args.embed_size, padding_idx=-1)         

        # _______ Scala Bias _______
        self.bias = nn.Parameter(torch.randn(1).normal_(0, 0.01), requires_grad=True)

        self.init_weight()

    def init_weight(self):
        self.u_emb.weight.data.normal_(0, 0.01)
        self.i_emb.weight.data.normal_(0, 0.01)
        self.cate_emb.weight.data.normal_(0, self.ip)
        self.attr_emb.weight.data.normal_(0, self.ip)
        self.cate_emb_context.weight.data.normal_(0, self.ip)
        self.attr_emb_context.weight.data.normal_(0, self.ip)
        self.cate_emb_target.weight.data.normal_(0, self.ip)
        self.attr_emb_target.weight.data.normal_(0, self.ip)

        # _______ set the padding to zero _______
        self.cate_emb.weight.data[-1,:] = 0
        self.attr_emb.weight.data[-1,:] = 0
        self.cate_emb_context.weight.data[-1,:] = 0
        self.attr_emb_context.weight.data[-1,:] = 0
        self.cate_emb_target.weight.data[-1,:] = 0
        self.attr_emb_target.weight.data[-1,:] = 0


    def forward(self, cates, attrs, cate_context, attr_context, users, items=None):

        u_emb = self.u_emb(users) # (bs, 1, emb_size)
        c_emb = self.cate_emb(cates) # (bs, max_len, emb_size)
        a_emb = self.attr_emb(attrs) # (bs, max_len, emb_size)

        c_emb_c = self.cate_emb_context(cate_context) # (bs, max_len, emb_size)
        a_emb_c = self.attr_emb_context(attr_context) # (bs, max_len, emb_size)

        if items is None:
            i_emb = self.i_emb.weight # (item_size, emb_size)
            cat_emb = torch.cat((u_emb, c_emb, a_emb, c_emb_c, a_emb_c), dim=1) # (bs, max_len+max_len_+1, emb_size)
            return torch.matmul(cat_emb, i_emb.T).sum(dim=1) # (bs, item_size)
        else:
            i_emb = self.i_emb(items) # (bs, 1, emb_size)
            cat_emb = torch.cat((u_emb, c_emb, a_emb, c_emb_c, a_emb_c), dim=1) # (bs, max_len+max_len_+1, emb_size)
            return torch.bmm(i_emb, cat_emb.transpose(1,2)).sum(dim=-1).sum(dim=-1) # (bs,)
