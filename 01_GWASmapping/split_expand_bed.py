import sys
import math

input_bed = sys.argv[1]
output_bed = sys.argv[2]

TARGET = 8000  # 8 kb

def write_interval(f, chrom, s, e):
    if s < 0:
        s = 0
    f.write(f"{chrom}\t{s}\t{e}\n")

with open(input_bed) as f, open(output_bed, "w") as out:
    for line in f:
        if line.strip() == "":
            continue
        parts = line.strip().split()
        chrom, start, end = parts[0], int(parts[1]), int(parts[2])

        L = end - start

        # Case 1: short intervals (L < 8k)
        if L < TARGET:
            expand = (TARGET - L) / 2
            new_start = int(start - expand)
            new_end = int(end + expand)
            write_interval(out, chrom, new_start, new_end)

        # Case 2: long intervals (L > 8k)
        else:
            cur = start
            # full 8k blocks
            while cur + TARGET <= end:
                write_interval(out, chrom, cur, cur + TARGET)
                cur += TARGET
            # remaining part
            remain = end - cur
            if remain > 0:
                # expand to 8k
                need = TARGET - remain
                new_end = end + need
                write_interval(out, chrom, cur, new_end)
