import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import math


compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {compute_device}")


# Read the dataset of 1000 generated Indian names
with open("TrainingNames.txt", "r", encoding="utf-8") as file_ptr:
    dataset_names = file_ptr.read().splitlines()

# Add special tokens for start-of-sequence (SOS) and end-of-sequence (EOS)
# We use '^' for SOS and '$' for EOS.
unique_chars = set(''.join(dataset_names))
unique_chars.add('^')
unique_chars.add('$')
char_vocabulary = sorted(list(unique_chars))
size_of_vocab = len(char_vocabulary)

# Create mapping dictionaries
c2i_map = {ch: idx for idx, ch in enumerate(char_vocabulary)}
i2c_map = {idx: ch for idx, ch in enumerate(char_vocabulary)}

print(f"Total names: {len(dataset_names)}")
print(f"Vocabulary size: {size_of_vocab} (Characters: {char_vocabulary})")

def convert_name_to_tensor(name_str):
    """
    Converts a string name into a tensor of integer indices, 
    wrapped with SOS and EOS tokens.
    """
    tensor_indices = [c2i_map['^']] + [c2i_map[ch] for ch in name_str] + [c2i_map['$']]
    return torch.tensor(tensor_indices, dtype=torch.long).to(compute_device)



class BasicRNNModel(nn.Module):
    def __init__(self, num_vocab, dim_embed, dim_hidden):
        super(BasicRNNModel, self).__init__()
        self.dim_hidden = dim_hidden
        
        # Embedding layer to convert character indices to dense vectors
        self.embed_layer = nn.Embedding(num_vocab, dim_embed)
        
        # RNN Cell Weights (Linear transformations for Input and Hidden states)
        # We avoid using nn.RNN here as per assignment constraints.
        self.weight_ih = nn.Linear(dim_embed, dim_hidden)  # Input to hidden
        self.weight_hh = nn.Linear(dim_hidden, dim_hidden) # Hidden to hidden
        
        # Output layer mapping hidden state back to vocabulary size
        self.linear_out = nn.Linear(dim_hidden, num_vocab)

    def forward(self, input_val, hidden_state):
        # input_val shape: (1) representing a single character index
        # hidden_state shape: (1, dim_hidden)
        
        # 1. Get embedding for the current character: (1, dim_embed)
        vec_embedded = self.embed_layer(input_val)
        
        # 2. Compute the new hidden state using the Vanilla RNN formula:
        # h_t = tanh(W_ih * x_t + W_hh * h_{t-1})
        hidden_state = torch.tanh(self.weight_ih(vec_embedded) + self.weight_hh(hidden_state))
        
        # 3. Compute the output prediction for the next character
        predicted_out = self.linear_out(hidden_state)
        
        return predicted_out, hidden_state

    def init_hidden(self):
        """Initializes the hidden state with zeros at the start of a sequence."""
        return torch.zeros(1, self.dim_hidden).to(compute_device)
    


class ManualLSTMCell(nn.Module):
    """A single LSTM cell implemented using raw linear layers and gates."""
    def __init__(self, size_input, size_hidden):
        super(ManualLSTMCell, self).__init__()
        self.size_hidden = size_hidden
        
        # The 4 gates of an LSTM: Input (i), Forget (f), Cell (g), Output (o)
        # We combine the input x_t and hidden h_{t-1} into one linear layer per gate for efficiency
        self.gate_i = nn.Linear(size_input + size_hidden, size_hidden)
        self.gate_f = nn.Linear(size_input + size_hidden, size_hidden)
        self.gate_g = nn.Linear(size_input + size_hidden, size_hidden)
        self.gate_o = nn.Linear(size_input + size_hidden, size_hidden)

    def forward(self, step_input, h_prev, c_prev):
        # Concatenate input and previous hidden state: [x_t, h_{t-1}]
        merged_input = torch.cat((step_input, h_prev), dim=1)
        
        # Gate calculations using standard LSTM equations
        val_i = torch.sigmoid(self.gate_i(merged_input))       # Input gate
        val_f = torch.sigmoid(self.gate_f(merged_input))       # Forget gate
        val_g = torch.tanh(self.gate_g(merged_input))          # Cell candidate
        val_o = torch.sigmoid(self.gate_o(merged_input))       # Output gate
        
        # Cell state update: c_t = f_t * c_{t-1} + i_t * g_t
        c_current = val_f * c_prev + val_i * val_g
        # Hidden state update: h_t = o_t * tanh(c_t)
        h_current = val_o * torch.tanh(c_current)
        
        return h_current, c_current

class BidirectionalLSTMModel(nn.Module):
    def __init__(self, num_vocab, dim_embed, dim_hidden):
        super(BidirectionalLSTMModel, self).__init__()
        self.dim_hidden = dim_hidden
        self.embed_layer = nn.Embedding(num_vocab, dim_embed)
        
        # Forward and Backward LSTM Cells
        self.cell_forward = ManualLSTMCell(dim_embed, dim_hidden)
        self.cell_backward = ManualLSTMCell(dim_embed, dim_hidden)
        
        # Output layer takes the concatenated forward and backward hidden states
        self.linear_out = nn.Linear(dim_hidden * 2, num_vocab)

    def forward(self, tensor_seq):
        length_seq = tensor_seq.size(0)
        emb_seq = self.embed_layer(tensor_seq) # (length_seq, dim_embed)
        
        # Initialize hidden and cell states for both directions
        h_fwd, c_fwd = self.init_states()
        h_bwd, c_bwd = self.init_states()
        
        states_fwd = []
        states_bwd = []
        
        # 1. Forward Pass (left to right)
        for step in range(length_seq):
            x_fwd = emb_seq[step].unsqueeze(0)
            h_fwd, c_fwd = self.cell_forward(x_fwd, h_fwd, c_fwd)
            states_fwd.append(h_fwd)
            
        # 2. Backward Pass (right to left)
        for step in range(length_seq - 1, -1, -1):
            x_bwd = emb_seq[step].unsqueeze(0)
            h_bwd, c_bwd = self.cell_backward(x_bwd, h_bwd, c_bwd)
            # Insert at beginning to maintain alignment with the forward sequence
            states_bwd.insert(0, h_bwd) 
            
        # 3. Concatenate and predict
        list_outputs = []
        for step in range(length_seq):
            # Concatenate forward and backward hidden states at time step t
            merged_h = torch.cat((states_fwd[step], states_bwd[step]), dim=1)
            pred_out = self.linear_out(merged_h)
            list_outputs.append(pred_out)
            
        return torch.cat(list_outputs, dim=0) # (length_seq, num_vocab)

    def init_states(self):
        """Initializes hidden and cell states with zeros."""
        return (torch.zeros(1, self.dim_hidden).to(compute_device), 
                torch.zeros(1, self.dim_hidden).to(compute_device))
    


class AttentionRNNModel(nn.Module):
    def __init__(self, num_vocab, dim_embed, dim_hidden):
        super(AttentionRNNModel, self).__init__()
        self.dim_hidden = dim_hidden
        self.embed_layer = nn.Embedding(num_vocab, dim_embed)
        
        # Standard RNN Cell Weights
        self.weight_ih = nn.Linear(dim_embed, dim_hidden)
        self.weight_hh = nn.Linear(dim_hidden, dim_hidden)
        
        # Attention Weights (Luang-style dot-product attention mapping)
        self.combine_attention = nn.Linear(dim_hidden * 2, dim_hidden)
        self.linear_out = nn.Linear(dim_hidden, num_vocab)

    def forward(self, input_val, hidden_state, outputs_encoder):
        # input_val: (1), hidden_state: (1, dim_hidden), outputs_encoder: (seq_len, dim_hidden)
        vec_embedded = self.embed_layer(input_val)
        
        # 1. Compute new hidden state (RNN step)
        hidden_state = torch.tanh(self.weight_ih(vec_embedded) + self.weight_hh(hidden_state))
        
        # 2. Calculate Attention Scores
        # Dot product between current hidden state and all past encoder outputs
        scores_attn = torch.matmul(hidden_state, outputs_encoder.transpose(0, 1))
        # Normalize scores to probabilities summing to 1
        scores_attn = F.softmax(scores_attn, dim=1) 
        
        # 3. Create context vector (weighted sum of past outputs)
        ctx_vector = torch.matmul(scores_attn, outputs_encoder) # (1, dim_hidden)
        
        # 4. Combine context vector with current hidden state
        merged_ctx = torch.cat((hidden_state, ctx_vector), dim=1)
        hidden_with_attn = torch.tanh(self.combine_attention(merged_ctx))
        
        # 5. Output prediction
        predicted_out = self.linear_out(hidden_with_attn)
        
        return predicted_out, hidden_state, scores_attn

    def init_hidden(self):
        return torch.zeros(1, self.dim_hidden).to(compute_device)




# Hyperparameters for vanilla rnn
VRNN_EMB_DIM = 64
VRNN_HID_DIM = 128
VRNN_LR = 0.001
VRNN_EPOCHS_NUM = 20

# Hyperparameters for blstm
BLSTM_EMB_DIM = 64
BLSTM_HID_DIM = 128
BLSTM_LR = 0.002
BLSTM_EPOCHS_NUM = 10

# Hyperparameters for attention rnn
ARNN_EMB_DIM = 64
ARNN_HID_DIM = 256
ARNN_LR = 0.001
ARNN_EPOCHS_NUM = 25

# Initialize models
net_rnn = BasicRNNModel(size_of_vocab, VRNN_EMB_DIM, VRNN_HID_DIM).to(compute_device)
net_blstm = BidirectionalLSTMModel(size_of_vocab, BLSTM_EMB_DIM, BLSTM_HID_DIM).to(compute_device)
net_attn = AttentionRNNModel(size_of_vocab, ARNN_EMB_DIM, ARNN_HID_DIM).to(compute_device)

#Trainable Parameters Function
def calculate_params(network_model):
    """Calculates the total number of trainable weights/biases in a model."""
    return sum(param.numel() for param in network_model.parameters() if param.requires_grad)

print("\n\n\n--- Architecture & Hyperparameter Report ---")
print(f"Hyperparameters -> Embed Size: {VRNN_EMB_DIM}, Hidden Size: {VRNN_HID_DIM}, LR: {VRNN_LR}")
print(f"1. Vanilla RNN Parameters: {calculate_params(net_rnn):,}")
print("-" * 50)

print(f"Hyperparameters -> Embed Size: {BLSTM_EMB_DIM}, Hidden Size: {BLSTM_HID_DIM}, LR: {BLSTM_LR}")
print(f"2. Bidirectional LSTM Parameters: {calculate_params(net_blstm):,}")
print("-" * 50)

print(f"Hyperparameters -> Embed Size: {ARNN_EMB_DIM}, Hidden Size: {ARNN_HID_DIM}, LR: {ARNN_LR}")
print(f"3. Attention RNN Parameters: {calculate_params(net_attn):,}")
print("-" * 50)



def sample_single_name(network_model, arch_type, length_max=20):
    """
    Generates a single name character by character using the trained model.
    Uses multinomial sampling to ensure diversity in the generated outputs.
    """
    network_model.eval()
    
    with torch.no_grad():
        # Start the sequence with the Start-Of-Sequence (SOS) token
        char_current = '^'
        name_generated = ""
        
        if arch_type == "vanilla":
            # Vanilla RNN only needs the hidden state carried over step-by-step
            state_hidden = network_model.init_hidden()
            
            for _ in range(length_max):
                # Convert current character string to a tensor index
                input_x = torch.tensor([c2i_map[char_current]], dtype=torch.long).to(compute_device)
                
                # Forward pass: predict next character logits and update hidden state
                pred_out, state_hidden = network_model(input_x, state_hidden)
                
                # Convert raw logits to probabilities
                probabilities = F.softmax(pred_out.squeeze(), dim=0)
                
                # Sample from the distribution instead of picking the absolute highest (argmax).
                idx_char = torch.multinomial(probabilities, 1).item()
                char_current = i2c_map[idx_char]
                
                # Stop generating if the model outputs the End-Of-Sequence (EOS) token
                if char_current == '$': break
                name_generated += char_current

        elif arch_type == "blstm":
            # BLSTMs are not naturally autoregressive. To generate text, we must 
            # feed it the entire generated sequence history at every single step.
            seq_prefix = [c2i_map['^']]
            
            for _ in range(length_max):
                input_x = torch.tensor(seq_prefix, dtype=torch.long).to(compute_device)
                all_outputs = network_model(input_x)
                
                # We only care about the prediction at the very last time step
                probabilities = F.softmax(all_outputs[-1].squeeze(), dim=0)
                
                # Sample the next character
                idx_char = torch.multinomial(probabilities, 1).item()
                char_current = i2c_map[idx_char]
                
                if char_current == '$': break
                name_generated += char_current
                
                # Append the newly generated character to the running sequence prefix
                seq_prefix.append(idx_char)

        elif arch_type == "attention":
            state_hidden = network_model.init_hidden()
            
            # Pre-allocate a blank memory matrix to store the hidden states of past generated characters.
            # The attention mechanism will look back at this matrix to decide what to generate next.
            outputs_enc = torch.zeros(length_max, network_model.dim_hidden).to(compute_device)
            
            for step_t in range(length_max):
                input_x = torch.tensor([c2i_map[char_current]], dtype=torch.long).to(compute_device)
                
                # Forward pass includes the growing history of encoder outputs
                pred_out, state_hidden, _ = network_model(input_x, state_hidden, outputs_enc)
                
                # Save the current hidden state into our memory matrix for future steps to attend to
                outputs_enc[step_t] = state_hidden.squeeze(0)
                
                probabilities = F.softmax(pred_out.squeeze(), dim=0)
                idx_char = torch.multinomial(probabilities, 1).item()
                char_current = i2c_map[idx_char]
                
                if char_current == '$': break
                name_generated += char_current
                
        return name_generated

def sample_multiple_names(network_model, arch_type, count_n=100):
    """
    Helper function to generate a batch of names by calling the single sampler repeatedly.
    """
    return [sample_single_name(network_model, arch_type) for _ in range(count_n)]



def compute_metrics(list_generated, list_training):
    """Calculates Novelty and Diversity metrics."""
    set_training = set(list_training)
    count_generated = len(list_generated)
    
    # Novelty: Names that are entirely new (not copied directly from training data)
    names_novel = [n for n in list_generated if n not in set_training]
    rate_novelty = len(names_novel) / count_generated
    
    # Diversity: How many unique names the model generated (avoids repeating the same name)
    names_unique = set(list_generated)
    rate_diversity = len(names_unique) / count_generated
    
    return rate_novelty * 100, rate_diversity * 100


print("\n\n\n--- Quantitative Evaluation (Generating 500 names each) ---")
EVAL_SAMPLES = 500

# List of models
models_list = [
    ("Vanilla RNN", net_rnn, "vanilla"),
    ("Bidirectional LSTM", net_blstm, "blstm"),
    ("Attention RNN", net_attn, "attention")
]

dict_generated_names = {}

for str_name, obj_model, str_type in models_list:
    
    # Load the saved best model weights from disk ---
    saved_file_name = f"best_{str_type}_model.pth"
    obj_model.load_state_dict(torch.load(saved_file_name, weights_only=True))
    obj_model.to(compute_device)
    obj_model.eval() # Set to evaluation mode
    
    # Generate names using the loaded best model
    sampled_names = sample_multiple_names(obj_model, str_type, count_n=EVAL_SAMPLES)
    dict_generated_names[str_name] = sampled_names
    
    val_novelty, val_diversity = compute_metrics(sampled_names, dataset_names)
    print(f"{str_name}:")
    print(f"  - Novelty Rate:   {val_novelty:.2f}%")
    print(f"  - Diversity Rate: {val_diversity:.2f}%\n")



print("\n\n\n--- Qualitative Samples ---")
for str_model_name, list_gen_names in dict_generated_names.items():
    print(f"\nModel: {str_model_name}")
    # Print 10 random samples from the generated pool
    random_samples = random.sample(list_gen_names, 10)
    for idx_samp, str_samp_name in enumerate(random_samples):
        print(f"  {idx_samp+1}. {str_samp_name}")