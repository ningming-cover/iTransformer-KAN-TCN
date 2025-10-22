import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import FFTransformer_EncDec as FFT
from layers.FFT_SelfAttention import FFTAttention
from layers.SelfAttention_Family import LogSparseAttentionLayer, ProbAttention, AttentionLayer, FullAttention
from layers.Embed import DataEmbedding
from layers.WaveletTransform import get_wt, get_fft, EMD_3D
from layers.Functionality import MLPLayer


class Model(nn.Module):
    """
    FFTransformer Encoder-Decoder with Convolutional ProbSparse Attn for Trend and ProbSparse for Freq Strean
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.output_attention = configs.output_attention
        self.num_decomp = configs.num_decomp

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Frequency Embeddings:
        self.enc_embeddingF = DataEmbedding(configs.enc_in*4, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, kernel_size=configs.kernel_size,
                                            temp_embed=False, pos_embed=False)
        self.dec_embeddingF = DataEmbedding(configs.dec_in*4, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, kernel_size=configs.kernel_size,
                                            temp_embed=False, pos_embed=False)

        # Trend Embeddings:
        self.enc_embeddingT = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, kernel_size=configs.kernel_size,
                                            temp_embed=False, pos_embed=False)
        self.dec_embeddingT = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, kernel_size=configs.kernel_size,
                                            temp_embed=False, pos_embed=False)

        self.encoder = FFT.Encoder(
            [
                FFT.EncoderLayer(
                    # FFTAttention(
                    #     ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                    #                   output_attention=configs.output_attention, top_keys=False, context_zero=True),
                    #     configs.d_model, configs.n_heads
                    # ),
                    # LogSparseAttentionLayer(
                    #     ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                    #                   output_attention=configs.output_attention, top_keys=configs.top_keys),
                    #     d_model=configs.d_model, n_heads=configs.n_heads,
                    #     qk_ker=configs.qk_ker, v_conv=configs.v_conv
                    # ),
                    # LogSparseAttentionLayer(
                    #     ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                    #                   output_attention=configs.output_attention, top_keys=configs.top_keys),
                    #     d_model=configs.d_model, n_heads=configs.n_heads,
                    #     qk_ker=configs.qk_ker, v_conv=configs.v_conv
                    # ),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
            norm_freq=torch.nn.LayerNorm(configs.d_model) if configs.norm_out else None,
            norm_trend=torch.nn.LayerNorm(configs.d_model) if configs.norm_out else None,
        )
        # Decoder
        self.decoder = FFT.Decoder(
            [
                FFT.DecoderLayer(
                    # FFTAttention(
                    #     ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                    #                   output_attention=configs.output_attention, top_keys=False, context_zero=True),
                    #     configs.d_model, configs.n_heads
                    # ),
                    # LogSparseAttentionLayer(
                    #     ProbAttention(True, configs.factor, attention_dropout=configs.dropout,
                    #                   output_attention=configs.output_attention, top_keys=configs.top_keys),
                    #     d_model=configs.d_model, n_heads=configs.n_heads, qk_ker=configs.qk_ker, v_conv=configs.v_conv
                    # ),
                    # FFTAttention(
                    #     ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                    #                   output_attention=configs.output_attention, top_keys=False, context_zero=True),
                    #     configs.d_model, configs.n_heads
                    # ),
                    # LogSparseAttentionLayer(
                    #     ProbAttention(True, configs.factor, attention_dropout=configs.dropout,
                    #                   output_attention=configs.output_attention, top_keys=configs.top_keys),
                    #     d_model=configs.d_model, n_heads=configs.n_heads, qk_ker=configs.qk_ker, v_conv=configs.v_conv
                    # ),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.d_layers)
            ],
            norm_freq=nn.LayerNorm(configs.d_model) if configs.norm_out else None,
            norm_trend=nn.LayerNorm(configs.d_model) if configs.norm_out else None,
            mlp_out=MLPLayer(d_model=configs.d_model, d_ff=configs.d_ff, kernel_size=1,
                             dropout=configs.dropout, activation=configs.activation) if configs.mlp_out else None,
            out_projection=nn.Linear(configs.d_model, configs.c_out),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, **_):
        # Wavelet Decomposition
        x_enc_freq, x_enc_trend = get_wt(x_enc, num_decomp=self.num_decomp)
        x_dec_freq, x_dec_trend = get_wt(x_dec[:, :-self.pred_len, :], num_decomp=self.num_decomp)  # Remove PHs first

        # Add placeholders after decomposition:
        dec_freq_place = torch.zeros([x_dec_freq.shape[0], self.pred_len, x_dec_freq.shape[2]], device=x_dec_freq.device)
        x_dec_freq = torch.cat([x_dec_freq, dec_freq_place], dim=1)

        dec_trend_place = torch.mean(x_dec_trend, 1).unsqueeze(1).repeat(1, self.pred_len, 1)
        x_dec_trend = torch.cat([x_dec_trend, dec_trend_place], dim=1)

        # Embed the inputs:
        x_enc_freq = self.enc_embeddingF(x_enc_freq, x_mark_enc)
        x_dec_freq = self.dec_embeddingF(x_dec_freq, x_mark_dec)

        x_enc_trend = self.enc_embeddingT(x_enc_trend, x_mark_enc)
        x_dec_trend = self.dec_embeddingT(x_dec_trend, x_mark_dec)

        # x_enc_period, x_enc_amplitude = get_fft(x_enc)
        # x_dec_period, x_dec_amplitude = get_fft(x_dec[:, :-self.pred_len, :])
        #
        # for i in range(3):
        #     if 64 % x_enc_period[i] != 0:
        #         x_enc_period[i] = 64
        #     if 48 % x_dec_period[i] != 0:
        #         x_dec_period[i] = 48
        #
        # aggregated_enc_period = []
        # aggregated_dec_period = []
        # X, Y, Z = x_enc.size()
        # B, L, d = x_dec[:, :-self.pred_len, :].size()
        #
        # for i in range(3):
        #     period = x_enc_period[i]
        #     out = x_enc.reshape(X, Y // period, period, Z)
        #     period_weight = F.softmax(x_enc_amplitude[:, i], dim=0)
        #
        #     # 将权重扩展为与数据相同的形状
        #     period_weight = period_weight.unsqueeze(1).unsqueeze(2).unsqueeze(-1).expand(B, L // period, period, d)
        #
        #     # 使用权重对数据进行加权求和
        #     aggregated_out = out * period_weight
        #
        #     # 形状变为(B, L, d)
        #     aggregated_out = torch.flatten(aggregated_out, start_dim=1, end_dim=2)
        #
        #     # 将结果添加到列表
        #     aggregated_enc_period.append(aggregated_out)
        # for i in range(3):
        #     period = x_dec_period[i]
        #     out = x_dec[:, :-self.pred_len, :].reshape(B, L // period, period, d)
        #     period_weight = F.softmax(x_dec_amplitude[:, i], dim=0)
        #
        #     # 将权重扩展为与数据相同的形状
        #     period_weight = period_weight.unsqueeze(1).unsqueeze(2).unsqueeze(-1).expand(B, L // period, period, d)
        #
        #     # 使用权重对数据进行加权求和
        #     aggregated_out = out * period_weight
        #
        #     # 形状变为(B, L, d)
        #     aggregated_out = torch.flatten(aggregated_out, start_dim=1, end_dim=2)
        #
        #     # 将结果添加到列表
        #     aggregated_dec_period.append(aggregated_out)
        # for i in range(3):
        #     # Add placeholders after the decomposition:
        #     period_place = torch.mean(aggregated_enc_period[i], 1).unsqueeze(1).repeat(1, self.pred_len, 1)
        #     aggregated_enc_period[i] = torch.cat([aggregated_enc_period[i], period_place], dim=1)
        # for i in range(3):
        #     # Add placeholders after the decomposition:
        #     period_place = torch.mean(aggregated_dec_period[i], 1).unsqueeze(1).repeat(1, self.pred_len, 1)
        #     aggregated_dec_period[i] = torch.cat([aggregated_dec_period[i], period_place], dim=1)
        # dec_trend_place = torch.mean(aggregated_dec_period[1], 1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # aggregated_dec_period[1] = torch.cat([aggregated_dec_period[1], dec_trend_place], dim=1)
        #
        # dec_freq_place = torch.zeros([aggregated_dec_period[0].shape[0], self.pred_len, aggregated_dec_period[0].shape[2]], device=aggregated_dec_period[0].device)
        # aggregated_dec_period[0] = torch.cat([aggregated_dec_period[0], dec_freq_place], dim=1)

        # Embed the inputs:
        # x_enc_freq = self.enc_embeddingF(aggregated_enc_period[0], x_mark_enc)
        # x_dec_freq = self.dec_embeddingF(aggregated_dec_period[0], x_mark_dec)
        #
        # x_enc_trend = self.enc_embeddingT(aggregated_enc_period[1], x_mark_enc)
        # x_dec_trend = self.dec_embeddingT(aggregated_dec_period[1], x_mark_dec)

        # imfs_enc = EMD_3D(x_enc, num_imfs=2)
        # x_enc_imf0 = imfs_enc[0].float()
        # x_enc_imf1 = imfs_enc[1].float()
        #
        # imfs_dec = EMD_3D(x_dec[:, :-self.pred_len, :], num_imfs=2)
        # x_dec_imf0 = imfs_dec[0].float()
        # x_dec_imf1 = imfs_dec[1].float()
        #
        # x_dec_imf0_place = torch.mean(x_dec_imf0, 1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # x_dec_imf0 = torch.cat([x_dec_imf0, x_dec_imf0_place], dim=1)
        #
        # x_dec_imf1_place = torch.mean(x_dec_imf1, 1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # x_dec_imf1 = torch.cat([x_dec_imf1, x_dec_imf1_place], dim=1)
        #
        # x_enc_imf0 = self.enc_embeddingF(x_enc_imf0, x_mark_enc)
        # x_enc_imf1 = self.enc_embeddingT(x_enc_imf1, x_mark_enc)
        #
        # x_dec_imf0 = self.dec_embeddingF(x_dec_imf0, x_mark_dec)
        # x_dec_imf1 = self.dec_embeddingT(x_dec_imf1, x_mark_dec)

        attns = []

        x_enc_freq, x_enc_trend, a = self.encoder([x_enc_freq, x_enc_trend], attn_mask=enc_self_mask)
        attns.append(a)

        dec_out, a = self.decoder([x_dec_freq, x_dec_trend], x_enc_freq, x_enc_trend, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        attns.append(a)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
