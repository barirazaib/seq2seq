# model.py
import torch
import torch.nn as nn
import math
import sentencepiece as spm

PAD_IDX = 0

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

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim * 2, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        emb = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(emb, (hidden, cell))
        pred = self.fc_out(output.squeeze(1))
        return pred, hidden, cell

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

def load_model(model_path, sp_model_path, emb_dim=256, hid_dim=256, layers=2, dropout=0.3, device="cpu"):
    """Load the trained model and tokenizer"""
    try:
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(sp_model_path):
            raise FileNotFoundError(f"Tokenizer file not found: {sp_model_path}")
        
        sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        vocab_size = sp.get_piece_size()

        encoder = Encoder(vocab_size, emb_dim, hid_dim, layers, dropout)
        decoder = Decoder(vocab_size, emb_dim, hid_dim, layers, dropout)
        model = Seq2Seq(encoder, decoder, device).to(device)

        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        return model, sp
        
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def transliterate_text(model, sentence, sp, device, max_len=50):
    """Transliterate input text using the trained model"""
    try:
        model.eval()
        with torch.no_grad():
            # Tokenize input
            tokens = [sp.bos_id()] + sp.encode(sentence, out_type=int) + [sp.eos_id()]
            src_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

            # Encode
            _, hidden_enc, cell_enc = model.encoder(src_tensor)
            dec_layers = model.decoder.rnn.num_layers
            hidden = model.transform_enc_to_dec(hidden_enc, dec_layers).to(device)
            cell   = model.transform_enc_to_dec(cell_enc, dec_layers).to(device)

            # Decode
            input_token = torch.tensor([sp.bos_id()], dtype=torch.long).to(device)
            outputs = []
            for _ in range(max_len):
                pred, hidden, cell = model.decoder(input_token, hidden, cell)
                top1 = pred.argmax(1).item()
                if top1 == sp.eos_id():
                    break
                outputs.append(top1)
                input_token = torch.tensor([top1], dtype=torch.long).to(device)

        return sp.decode(outputs)
    except Exception as e:
        raise Exception(f"Error during transliteration: {str(e)}")
