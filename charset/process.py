
fout = open("./chinese_common.txt","w")

with open("./chinese_common_500_raw.txt", "r") as fin:
    chars = set(fin.read())
    chars -= {" ", "\n"}

chars = list(chars)
chars = sorted(chars, key=ord)

for char in chars:
    fout.write(char)