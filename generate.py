import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from model import Model, EMB_DIM, FFN_DIM, NUM_LAYERS, SEQ_LEN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = 'model.pth'

MAX_NEW_TOKENS = 128
TEMPERATURE = 0
TOP_K = 50
TOP_P = 0.9
REPETITION_PENALTY = 1.1

tok = Tokenizer.from_file('tokenizer.json')
tok.decoder = ByteLevelDecoder()

model = Model(
    EMB_DIM,
    tok.get_vocab_size(),
    NUM_LAYERS,
    SEQ_LEN,
    FFN_DIM
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint)

model.eval()


def repetition_penalty(logits, ids, penalty):
    if penalty == 1.0:
        return logits

    for token_id in set(ids):
        if logits[token_id] < 0:
            logits[token_id] *= penalty
        else:
            logits[token_id] /= penalty

    return logits


def top_k(logits, k):
    if k <= 0:
        return logits

    k = min(k, logits.numel())

    values = torch.topk(logits, k).values
    threshold = values[-1]

    logits[logits < threshold] = -float('inf')

    return logits


def top_p(logits, p):
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True
    )

    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)

    remove = cumulative > p

    remove[1:] = remove[:-1].clone()
    remove[0] = False

    sorted_logits[remove] = -float('inf')

    logits = torch.full_like(logits, -float('inf'))
    logits.scatter_(0, sorted_indices, sorted_logits)

    return logits


@torch.no_grad()
def generate(prompt):
    encoded = tok.encode(prompt)
    generated = encoded.ids.copy()

    eos_id = tok.token_to_id('<EOS>')

    for _ in range(MAX_NEW_TOKENS):

        context = generated[-SEQ_LEN:]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=DEVICE
        )

        logits = model(x)

        next_logits = logits[0, -1].clone()

        

        if TEMPERATURE <= 0:
            next_token = torch.argmax(next_logits).item()

        else:
            next_logits = next_logits / TEMPERATURE

            next_logits = top_k(
                next_logits,
                TOP_K
            )

            next_logits = top_p(
                next_logits,
                TOP_P
            )

            probabilities = torch.softmax(
                next_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                1
            ).item()

        generated.append(next_token)

        if eos_id is not None and next_token == eos_id:
            break

    return tok.decode(generated)


print('Model loaded.')
print('Type "exit" to quit.')

while True:

    prompt = input('\nPrompt: ')

    if prompt.lower() in ['exit', 'quit']:
        break

    output = generate(prompt)

    print('\n' + output)