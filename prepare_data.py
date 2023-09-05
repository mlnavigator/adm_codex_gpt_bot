import re
import json
import os

# Download kodap here https://www.kodap.ru/skachat-koap-rf and copy to assets/kodap.txt

base_dir = os.path.dirname(os.path.abspath(__file__))

fn = os.path.join(base_dir, 'assets/kodap.txt')

with open(fn, 'r') as f:
    data = f.read()
    
data = data.replace('\nСтатья', '\n<sep>Статья')

data = data.split('<sep>')

data = [d.strip() for d in data if 'Статья' in d]


def clear_text(t):
    t = str(t)
    t = t.replace('\xa0', ' ')
    tt = t.split('\n')
    res = '\n'.join([s for s in tt if not s.startswith('Глава ') and len(s.strip()) > 0])
    return res


data = [clear_text(d) for d in data]


def get_name(t):
    try:
        name_full = t.split('\n')[0]
        name_short = re.search('Статья \d+\.\d+\.{0,1}\d*', name_full).group(0)
        if name_short.endswith('.'):
            name_short = name_short[:-1]
    except Exception as e:
        print(t)
        raise e
    return name_short, name_full, t


data = [get_name(d) for d in data]

with open(os.path.join(base_dir, 'assets/data.json'), 'w') as f:
    json.dump(data, f, ensure_ascii=False)

print(len(data), 'data ready')

print('start vectoryzing')

import vectoryze

