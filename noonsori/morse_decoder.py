from typing import List


def _norm(morse: str) -> str:
    return (morse.replace('•', '.').replace('●', '.').replace('·', '.')
            .replace('–', '-').replace('—', '-').replace('ー', '-')
            .replace('_', '-')).strip()


HANGUL_MORSE = {
    'ㄱ': '.-..', 'ㄴ': '..-.', 'ㄷ': '-...', 'ㄹ': '...-', 'ㅁ': '--',
    'ㅂ': '.--', 'ㅅ': '--.', 'ㅇ': '-.-', 'ㅈ': '.--.', 'ㅊ': '-.-.',
    'ㅋ': '-..-', 'ㅌ': '--..', 'ㅍ': '---', 'ㅎ': '.---',
    'ㅏ': '.', 'ㅑ': '..', 'ㅓ': '-', 'ㅕ': '...', 'ㅗ': '.-', 'ㅛ': '-.',
    'ㅜ': '....', 'ㅠ': '.-.', 'ㅡ': '-..', 'ㅣ': '..-', 'ㅐ': '--.-', 'ㅔ': '-.--'
}
DIGIT_MORSE = {'1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
               '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----'}
MORSE_TO_DIGIT = {v: k for k, v in DIGIT_MORSE.items()}
MORSE_TO_HANGUL = {v: k for k, v in HANGUL_MORSE.items()}

DOUBLE_MAP = {'ㄱ': 'ㄲ', 'ㄷ': 'ㄸ', 'ㅂ': 'ㅃ', 'ㅅ': 'ㅆ', 'ㅈ': 'ㅉ'}
COMPOSE_VOWELS = {
    ('ㅕ', 'ㅣ'): 'ㅖ', ('ㅑ', 'ㅣ'): 'ㅒ',
    ('ㅏ', 'ㅣ'): 'ㅐ', ('ㅓ', 'ㅣ'): 'ㅔ',
    ('ㅗ', 'ㅏ'): 'ㅘ', ('ㅗ', 'ㅐ'): 'ㅙ', ('ㅗ', 'ㅣ'): 'ㅚ',
    ('ㅜ', 'ㅓ'): 'ㅝ', ('ㅜ', 'ㅔ'): 'ㅞ', ('ㅜ', 'ㅣ'): 'ㅟ',
    ('ㅡ', 'ㅣ'): 'ㅢ',
}


def _compose_vowels(stream: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(stream):
        # 3자 패턴
        if i + 2 < len(stream):
            pair12 = (stream[i + 1], stream[i + 2])
            if pair12 in COMPOSE_VOWELS:
                mid = COMPOSE_VOWELS[pair12]
                pair1 = (stream[i], mid)
                if pair1 in COMPOSE_VOWELS:
                    out.append(COMPOSE_VOWELS[pair1])
                    i += 3
                    continue
        # 2자 패턴
        if i + 1 < len(stream):
            pair = (stream[i], stream[i + 1])
            if pair in COMPOSE_VOWELS:
                out.append(COMPOSE_VOWELS[pair])
                i += 2
                continue
        # 합성 불가 → 그대로
        out.append(stream[i])
        i += 1
    return out


def decode_morse_to_korean(morse_line: str) -> str:
    s = _norm(morse_line)
    raw_words: List[List[str]] = []
    for chunk in s.split('/'):
        parts = [p for p in chunk.strip().split(' ') if p]
        if parts: raw_words.append(parts)

    decoded_words: List[str] = []
    for tokens in raw_words:
        jamo_seq: List[str] = []
        i = 0
        while i < len(tokens):
            code = tokens[i]

            # 숫자
            if code in MORSE_TO_DIGIT:
                jamo_seq.append(MORSE_TO_DIGIT[code])
                i += 1
                continue

            # 한글 자모
            if code in MORSE_TO_HANGUL:
                base = MORSE_TO_HANGUL[code]
                # 같은 코드가 연속 두 번 → 된소리
                if i + 1 < len(tokens) and tokens[i + 1] == code and base in DOUBLE_MAP:
                    jamo_seq.append(DOUBLE_MAP[base])
                    i += 2
                else:
                    jamo_seq.append(base)
                    i += 1
                continue

            # 알 수 없는 코드는 그대로 표시
            jamo_seq.append(f'[{code}]')
            i += 1

            # 모음 합성(이중모음)
        jamo_seq = _compose_vowels(jamo_seq)
        decoded_words.append(''.join(jamo_seq))

    return ' '.join(decoded_words)
