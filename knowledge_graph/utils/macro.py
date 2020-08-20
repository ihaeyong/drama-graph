import json, os

# 디렉토리 안의 모든 파일에 대한 absolute path return
def diriter(path):
    for p, d, f in os.walk(path):
        for ff in f:
            yield "/".join([p, ff])


def readfile(path):
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            yield line.strip()


# 파일 이름으로 json 로드(utf-8만 해당)
def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)
    return j

# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
	with open(fname, "w", encoding="UTF8") as f:
		json.dump(j, f, ensure_ascii=False, indent="\t")