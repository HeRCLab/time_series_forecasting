import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear


def smape_loss(y_pred, target):
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


# NO_CHANNELS = 512
NO_CHANNELS = 16

# NO_HEAD = 8
NO_HEAD = 1
# NO_LAYERS = 8
NO_LAYERS = 1

# ifty
# from v1 tutorial
# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: Tensor) -> Tensor:
#         # Arguments:
#         #     x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)


class TimeSeriesForcasting(pl.LightningModule):
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        no_unique_vals_in_data,
        channels=NO_CHANNELS,
        dropout=0.1,
        lr=1e-4,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.dropout = dropout
        # self.no_unique_vals_in_data = no_unique_vals_in_data

        # self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        # self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

        # self.input_pos_embedding = torch.nn.Embedding(no_unique_vals_in_data, embedding_dim=channels)
        # self.target_pos_embedding = torch.nn.Embedding(no_unique_vals_in_data, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=NO_HEAD,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=NO_HEAD,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=NO_LAYERS)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=NO_LAYERS)

        self.input_projection = Linear(n_encoder_inputs, channels)
        print('self.input_projection = ', self.input_projection)
        self.output_projection = Linear(n_decoder_inputs, channels)
        print('self.output_projection = ', self.output_projection)

        self.linear = Linear(channels, 1)

        self.do = nn.Dropout(p=self.dropout)
        print('check ifty')

    def encode_src(self, src):
        print('@encode_src 1. src = ', src)

        src_start = self.input_projection(src) #.permute(1, 0, 2)

        print('@encode_src 1. src_start = ', src_start)

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        print('@encode_src 1. in_sequence_len = ', in_sequence_len)
        print('@encode_src 1. batch_size = ', batch_size)


        # pos_encoder = (
        #     torch.arange(0, in_sequence_len, device=src.device)
        #     .unsqueeze(0)
        #     .repeat(batch_size, 1)
        # )
        # print('@encode_src 1. pos_encoder = ', pos_encoder)
        #
        # pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)
        # print('@encode_src 1.2 pos_encoder = ', pos_encoder)

        # src = src_start + pos_encoder
        # print('@encode_src 1.2 src = ', src)

        src = self.encoder(src) + src_start
        # print('@encode_src 1.2.2 src = ', src)

        print('@encode_src 2. src = ', src)
        return src

    def decode_trg(self, trg, memory):

        trg_start = self.output_projection(trg) #.permute(1, 0, 2)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)
        print('@decode_trg 1. out_sequence_len = ', out_sequence_len)
        print('@decode_trg 1. batch_size = ', batch_size)
        # pos_decoder = (
        #     torch.arange(0, out_sequence_len, device=trg.device)
        #     .unsqueeze(0)
        #     .repeat(batch_size, 1)
        # )
        # pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)
        #
        # trg = pos_decoder + trg_start

        # trg_mask = gen_trg_mask(out_sequence_len, trg.device)
        trg_mask = gen_trg_mask(out_sequence_len, trg.device)

        print('@decode_trg trg_mask.size() = ', trg_mask.size())
        print('@decode_trg trg.size() = ', trg.size())

        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start

        out = out.permute(1, 0, 2)

        out = self.linear(out)
        print('@decode_trg out = ', out)

        return out

    def forward(self, x):
        print()
        # print('@forward x = ', x)
        src, trg = x
        # print('@forward src = ', src)
        print('@forward src.size() = ', src.size())
        print('@forward trg.size() = ', trg.size())

        # print('@forward trg = ', trg)


        src = self.encode_src(src)
        print('@forward after self.encode_src(src) src = ', src)

        out = self.decode_trg(trg=trg, memory=src)
        print('@forward after self.decode_trg(trg=trg, memory=src) out = ', out)

        return out

    def training_step(self, batch, batch_idx):
        # print('@training_step batch = ', batch)
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))
        # print('@training_step y_hat = ', y_hat)
        y_hat = y_hat.view(-1)
        print('@training_step after y_hat.view(-1) y_hat = ', y_hat)

        y = trg_out.view(-1)
        print('@training_step after trg_out.view(-1) y = ', y)

        loss = smape_loss(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }


if __name__ == "__main__":
    n_classes = 100

    source = torch.rand(size=(32, 16, 9))
    target_in = torch.rand(size=(32, 16, 8))
    target_out = torch.rand(size=(32, 16, 1))

    ts = TimeSeriesForcasting(n_encoder_inputs=9, n_decoder_inputs=8)

    print('@__main__ model source = ', source)
    print('@__main__ model target_in = ', target_in)

    pred = ts((source, target_in))

    print('@__main__ model pred = ', pred)


    print('pred.size() = ', pred.size())

    ts.training_step((source, target_in, target_out), batch_idx=1)
