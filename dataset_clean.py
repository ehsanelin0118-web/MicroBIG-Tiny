import regex
import unicodedata
import os

RE_URL = regex.compile(r'https?://\S+|www\.\S+', regex.I)
RE_EMAIL = regex.compile(r'[\p{L}\p{N}._%+\-]+@[\p{L}\p{N}.\-]+\.[\p{L}]{2,}')
RE_CONTROL = regex.compile(r'[\p{Cc}--[\r\n\t]]')
RE_ZERO_WIDTH = regex.compile(r'[\u200B-\u200D\uFEFF]')

RE_CHAR_RUN = regex.compile(
    r'(?P<char>[^\W\s_])(?P=char){3,}',
    regex.I
)

RE_WORD_RUN = regex.compile(
    r'\b(?P<word>[\p{L}\p{N}]+)(?:[ \t]+(?P=word)){1,}\b',
    regex.I
)

RE_LONG_WORD = regex.compile(
    r'\b(?P<base>[\p{L}\p{N}]{3,}?)(?P<char>[\p{L}])(?P=char){3,}\b',
    regex.I
)

RE_PUNCT_RUN = regex.compile(
    r'(?P<punc>[!?.,:;])(?P=punc){2,}'
)

RE_DASH_RUN = regex.compile(r'[-‐-‒–—]{2,}')
RE_SPACE = regex.compile(r'[^\S\r\n]+')
RE_NEWLINES = regex.compile(r'\n{3,}')
RE_SPACE_BEFORE_PUNCT = regex.compile(r'[ \t]+([!?.,:;])')
RE_SPACE_AFTER_PUNCT = regex.compile(r'([!?.,:;])[ \t]+')
RE_EMPTY_LINES = regex.compile(r'[ \t]+\n')


def clean_text(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    text = unicodedata.normalize("NFKC", text)

    text = RE_ZERO_WIDTH.sub('', text)
    text = RE_CONTROL.sub('', text)

    text = RE_URL.sub('', text)
    text = RE_EMAIL.sub('', text)

    text = RE_LONG_WORD.sub(
        lambda m: m.group('base') + m.group('char'),
        text
    )

    text = RE_CHAR_RUN.sub(
        lambda m: m.group('char') * 2,
        text
    )

    text = RE_WORD_RUN.sub(
        lambda m: m.group('word'),
        text
    )

    text = RE_PUNCT_RUN.sub(
        lambda m: m.group('punc'),
        text
    )

    text = RE_DASH_RUN.sub('-', text)

    text = RE_SPACE_BEFORE_PUNCT.sub(r'\1', text)
    text = RE_SPACE_AFTER_PUNCT.sub(r'\1 ', text)

    text = RE_SPACE.sub(' ', text)
    text = RE_EMPTY_LINES.sub('\n', text)
    text = RE_NEWLINES.sub('\n\n', text)
    text = regex.sub(r'<unk>', ' ', text, flags=regex.I)

    return text.strip()

files = os.listdir('datasets')
with open('datasets/cleaned.txt', 'w', encoding='utf-8') as cleaned_file:
    for file in files:
        if file.endswith('.txt') and file != 'cleaned.txt':
            with open(os.path.join('datasets', file), 'r', encoding='utf-8') as f:
                text = f.read()
                
            cleaned_text = clean_text(text)
            cleaned_file.write(cleaned_text + '\n\n')
