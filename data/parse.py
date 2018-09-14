"""
Copyright (c) Zacker
Hawk I
"""

import os
from urllib import parse
import re
import json

class ParseData:
    def __init__(self, fs):
        assert fs in ['n1', 'n2', 'a'], 'File Symbol not found.'
        curpath = os.path.dirname(os.path.realpath(__file__))
        if os.path.exists(f'{curpath}/CSIC2010/{fs}.json'):
            with open(f'{curpath}/CSIC2010/{fs}.json', 'r') as infile:
                self.samples = json.loads(infile.readline())
        else:
            files = {}
            files['n1'] = f'{curpath}/CSIC2010/normalTrafficTraining.txt'
            files['a'] = f'{curpath}/CSIC2010/anomalousTrafficTest.txt'
            files['n2'] = f'{curpath}/CSIC2010/normalTrafficTest.txt'
            self.samples = self.get_samples(files[fs])

    def get_samples(self, file):
        assert os.path.exists(file), 'File does not exist.'
        with open(file, 'r') as f:
            lines = f.readlines()

        samples = []
        sample = {}
        for line in lines:
            if (line.startswith('GET ') or line.startswith('POST ') or line.startswith('PUT ')):
                if sample:
                    # concat get param and post/put param
                    for i in ['Ori', 'Norm']:
                        ps = {}
                        for t in [f'Url{i}Params', f'Body{i}Params']:
                            if t in sample.keys():
                                for k, v in sample[t].items():
                                    ps[k] = v
                                del sample[t]
                        sample[f'{i}Params'] = ps
                    samples.append(sample)
                sample = {}
            line = line.strip('\n')
            if line:
                #First line
                if sample == {}:
                    parts = line.split(' ')
                    sample['Method'] = parts[0]
                    parsed = parse.urlparse(parts[1])
                    sample['Path'] = parsed.path
                    sample['UrlNormParams'], sample['UrlOriParams'] = self.parse_query(parsed.query)
                elif ': ' not in line:
                    sample['BodyNormParams'], sample['BodyOriParams'] = self.parse_query(line)
        return samples

    def normalize(self, s, with_sub=True):
        #urldecode
        while True:
            new_s = parse.unquote(s, encoding='ascii', errors='ignore')
            if new_s == s:
                break
            else:
                s = new_s
        #normalize
        if with_sub:
            s = re.sub('\\ufffd', 'a', s)
            s = re.sub('[a-zA-Z]', 'a', s)
            s = re.sub('\d', 'n', s)
            s = re.sub('a+', 'a+', s)
            s = re.sub('n+', 'n+', s)
            s = re.sub(' ', '_', s)
        return s

    def parse_query(self, query):
        raw_params = parse.parse_qs(query)
        origin_params = {}
        norm_params = {}
        for k, v in raw_params.items():
            norm_params[k] = self.normalize(v[0], with_sub=True)
            origin_params[k] = self.normalize(v[0], with_sub=False)
        return norm_params, origin_params

    def set_key(self, sample, kv):
        if kv[0] in sample.keys():
            self.set_key(sample, [kv[0]+'*', kv[1]])
        else:
            if kv[0].startswith('Cookie'):
                sample[kv[0]] = self.extract_cookies(kv[1])
            else:
                sample[kv[0]] = kv[1]

    def extract_cookies(self, cookie):
        cookies = dict([l.strip(' ').split("=", 1) for l in cookie.split(";")])
        return cookies

if __name__ == '__main__':
    print('#Parsing Original Dataset.')
    for fn in ['n1', 'n2', 'a']:
        pd = ParseData(fn)
        with open(f'CSIC2010/{fn}.json', 'w') as outfile:
            print(f'#Sample count in {fn}: {len(pd.samples)}')
            outfile.write(json.dumps(pd.samples))
            print(f'#{fn}.json generated.')
    print('#Congratulations! Parsing Process Done. Enjoy JSON!')