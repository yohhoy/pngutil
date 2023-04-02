#!/usr/bin/env python3
"""
PNG utilities tool
Copyright (c) 2023 yohhoy
"""
import argparse
import struct


PNG_SIG = b'\x89PNG\x0d\x0a\x1a\x0a'


def crc_lut(n):
    for _ in range(8):
        n = 0xedb88320 ^ (n >> 1) if n & 1 else n >> 1
    return n
CRC_TABLE = [crc_lut(n) for n in range(256)]


def calc_crc(data):
    def update_crc(c):
        for b in data:
            c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >> 8)
        return c
    return update_crc(0xffffffff) ^ 0xffffffff


# read PNG chunk
def read_chunk(f, args):
    size, ctype = struct.unpack('>I4s', f.read(8))
    discard = ' <discard>' if (ctype not in args.keep) else ''
    print(f'{ctype.decode("ascii")}: length={size}{discard}')
    data = f.read(size)
    crc = struct.unpack('>I', f.read(4))[0]
    assert crc == calc_crc(ctype + data)
    return (ctype, data)


# write PNG chunk
def write_chunk(f, ctype, data):
    assert len(ctype) == 4
    f.write(struct.pack('>I', len(data)))
    f.write(ctype)
    f.write(data)
    f.write(struct.pack('>I', calc_crc(ctype + data)))


# parse image header(IHDR) chunk
def parse_IHDR(chunk):
    assert chunk[0] == b'IHDR'
    K = ('width', 'height', 'bitdepth', 'color', 'compression', 'filter', 'interlace')
    ihdr = dict(zip(K, struct.unpack('>IIBBBBB', chunk[1])))
    for k in K:
        print(f'  {k}={ihdr[k]}')
    return ihdr


# parse primary chromaticities and white point(cHRM) chunk
def parse_cHRM(chunk, ihdr):
    assert chunk[0] == b'cHRM'
    wx, wy, rx, ry, gx, gy, bx, by = struct.unpack('>IIIIIIII', chunk[1])
    print(f'  White point x={wx/100000} ({wy})')
    print(f'  White point y={wy/100000} ({wy})')
    print(f'  Red x={rx/100000} ({rx})')
    print(f'  Red y={ry/100000} ({ry})')
    print(f'  Green x={gx/100000} ({gx})')
    print(f'  Green y={gy/100000} ({gy})')
    print(f'  Blue x={bx/100000} ({bx})')
    print(f'  Blue y={by/100000} ({by})')


# parse textual data(tEXt) chunk
def parse_tEXt(chunk, ihdr):
    assert chunk[0] == b'tEXt'
    key, _, text = chunk[1].partition(b'\x00')
    print(f'  Keyword="{key.decode("latin-1")}"')
    print(f'  Text="{text.decode("latin-1")}"')


# parse background colour(bKGD) chunk
def parse_bKGD(chunk, ihdr):
    assert chunk[0] == b'bKGD'
    ct = ihdr['color']
    if ct == 0 or ct == 4:
        bg = struct.unpack('>H', chunk[1])
        print(f'  Grayscale={bg}')
    if ct == 2 or ct == 6:
        br, bg, bb = struct.unpack('>HHH', chunk[1])
        print(f'  Red={br}')
        print(f'  Green={bg}')
        print(f'  Blue={bb}')
    if ct == 3:
        pi = struct.unpack('>B', chunk[1])
        print(f'  Palette index={pi}')


# process PNG format
def process_png(fin, fout, args):
    signature = fin.read(8)
    assert signature == PNG_SIG, 'Input file is not PNG format'
    fout.write(PNG_SIG)
    # read PNG chunks (with filter)
    png, ihdr = [], None
    parser = {b'cHRM': parse_cHRM, b'bKGD': parse_bKGD, b'tEXt': parse_tEXt}
    while True:
        chunk = read_chunk(fin, args)
        if chunk[0] == b'IHDR':
            ihdr = parse_IHDR(chunk)
        elif chunk[0] == b'IEND':
            break
        if chunk[0] in parser.keys():
            parser[chunk[0]](chunk, ihdr)
        png.append(chunk)
    # merge IDAT chunks
    if args.merge_idat:
        idat_chunks = [chunk[1] for chunk in png if chunk[0] == b'IDAT']
        if len(idat_chunks) > 1:
            idat = bytearray().join(idat_chunks)
            print(f'Merge: {len(idat_chunks)} IDAT chunks, {len(idat)} bytes')
            new_png = []
            for chunk in png:
                if chunk[0] == b'IDAT':
                    if idat:
                        new_png.append((b'IDAT', idat))
                        idat = None
                else:
                    new_png.append(chunk)
            png = new_png
    # write PNG chunks
    for chunk in png:
        write_chunk(fout, *chunk)


def main(args):
    print(f'Input: {args.infile}')
    print(f'Output: {args.outfile}')
    # process PNG format
    print(f'Filter: keep={args.keep}')
    print(f'Option: merge-idat={args.merge_idat}')
    args.keep = [bytes(ct, 'ascii') for ct in args.keep]
    with open(args.infile, 'rb') as fin, open(args.outfile, 'wb') as fout:
        process_png(fin, fout, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('pngutil')
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('--keep', action='append', metavar='CHUNK',
                        default=['IHDR', 'IDAT', 'PLTE', 'IEND'],
                        help='keep PNG chunk (%(default)s)')
    parser.add_argument('--merge-idat', metavar='0|1', default=1,
                        help='merge to single IDAT chunk (default: %(default)s)')
    args = parser.parse_args()
    main(args)