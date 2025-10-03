# model.py
import torch
import torch.nn as nn
import sentencepiece as spm
import math

PAD_IDX = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        emb = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(emb)
        return outputs, hidden, cell

# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim*2, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hid_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        emb = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(emb, (hidden, cell))
        pred = self.fc_out(output.squeeze(1))
        return pred, hidden, cell

# -------------------------
# Seq2Seq Model
# -------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    @staticmethod
    def transform_enc_to_dec(state, dec_layers):
        enc_layers = state.size(0) // 2
        pieces = []
        for i in range(enc_layers):
            f = state[2*i]
            b = state[2*i+1]
            pieces.append(torch.cat((f, b), dim=1))
        new_state = torch.stack(pieces, dim=0)
        if new_state.size(0) < dec_layers:
            reps = math.ceil(dec_layers / new_state.size(0))
            new_state = new_state.repeat(reps, 1, 1)
        return new_state[:dec_layers]

# -------------------------
# Load model + tokenizer
# -------------------------
def load_model(model_path="best_seq2seq_joint.pth", sp_model="joint_char.model",
               emb_dim=256, hid_dim=256, layers=2, dropout=0.3):
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    vocab_size = sp.get_piece_size()

    encoder = Encoder(vocab_size, emb_dim, hid_dim, layers, dropout)
    decoder = Decoder(vocab_size, emb_dim, hid_dim, layers, dropout)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, sp

# -------------------------
# Inference
# -------------------------
def transliterate_text(model, sentence, sp, max_len=50):
    model.eval()
    with torch.no_grad():
        tokens = [sp.bos_id()] + sp.encode(sentence, out_type=int) + [sp.eos_id()]
        src_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

        _, hidden_enc, cell_enc = model.encoder(src_tensor)
        dec_layers = model.decoder.rnn.num_layers
        hidden = model.transform_enc_to_dec(hidden_enc, dec_layers).to(DEVICE)
        cell   = model.transform_enc_to_dec(cell_enc, dec_layers).to(DEVICE)

        input_token = torch.tensor([sp.bos_id()], dtype=torch.long).to(DEVICE)
        outputs = []
        for _ in range(max_len):
            pred, hidden, cell = model.decoder(input_token, hidden, cell)
            top1 = pred.argmax(1).item()
            if top1 == sp.eos_id():
                break
            outputs.append(top1)
            input_token = torch.tensor([top1], dtype=torch.long).to(DEVICE)
    return sp.decode(outputs)
