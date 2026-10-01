from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC

VOCAB_SIZE = 8192

tok = Tokenizer(BPE(unk_token='<UNK>'))

tok.normalizer = NFKC()
tok.pre_tokenizer = ByteLevel()
tok.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    show_progress=True,
    special_tokens=[
        '<PAD>',
        '<UNK>',
        '<USER>',
        '<ROBOT>',
        '<EOS>',
        '<BOS>'
    ]
)

tok.train(
    ['datasets/cleaned.txt'],
    trainer
)

tok.save("tokenizer.json")