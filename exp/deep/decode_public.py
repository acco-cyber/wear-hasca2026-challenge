import json, base64, io, zipfile, re
nb = json.load(open(r'E:\Claude code\wear\kernels\honghanhhh__wear-hasca-hungarian-chain-viterbi-lb-0-74\wear-hasca-hungarian-chain-viterbi-lb-0-74.ipynb', encoding='utf-8'))
s = ''.join(nb['cells'][4]['source'])
m = re.search(r'PAYLOAD = """(.*?)"""', s, re.S)
b = base64.b64decode(m.group(1).replace('\n', ''))
z = zipfile.ZipFile(io.BytesIO(b))
out = r'E:\Claude code\wear\exp\deep\public_src'
z.extractall(out)
for n in z.namelist():
    print(n, z.getinfo(n).file_size)
print(''.join(nb['cells'][8]['source']))
