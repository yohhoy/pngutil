#!/usr/bin/env python3
"""
PNG utilities tool
Copyright (c) 2023 yohhoy
"""
import argparse
import binascii
import math
import struct
import sys
import zlib


PNG_SIG = b'\x89PNG\x0d\x0a\x1a\x0a'


# read PNG chunk
def read_chunk(f, args):
    size, ctype = struct.unpack('>I4s', f.read(8))
    discard = ' <discard>' if (ctype not in args.keep) else ''
    print(f'{ctype.decode("ascii")}: length={size}{discard}')
    data = f.read(size)
    crc = struct.unpack('>I', f.read(4))[0]
    assert crc == binascii.crc32(ctype + data)
    return (ctype, data)


# write PNG chunk
def write_chunk(f, ctype, data):
    assert len(ctype) == 4
    f.write(struct.pack('>I', len(data)))
    f.write(ctype)
    f.write(data)
    f.write(struct.pack('>I', binascii.crc32(ctype + data)))


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


# encode zlib/deflate 'non-compressed blocks' per ONE byte
def encode_noncompress(data):
    stream = bytearray(b'\x78\x01')   
    for b in data[:-1]:
        stream += b'\x00\x01\x00\xfe\xff' # BFINAL=0 BTYPE=00
        stream.append(b)
    stream += b'\x01\x01\x00\xfe\xff' # BFINAL=1 BTYPE=00
    stream.append(data[-1])
    stream += struct.pack('>I', zlib.adler32(data))
    return stream


# recompress IDAT data
def recompress_idat(png, args):
    if args.recompress == -1 and not args.bloat:
        return png
    idat_idx = [idx for idx, chunk in enumerate(png) if chunk[0] == b'IDAT']
    before_idat = [chunk for idx, chunk in enumerate(png) if idx < idat_idx[0]]
    after_idat = [chunk for idx, chunk in enumerate(png) if idat_idx[0] < idx and idx not in idat_idx]
    if len(idat_idx) > 1:
        print(f'Recompress: Input PNG has multiple IDAT chunks; Use "--merge-idat 1".')
        sys.exit(1)
    # recompress zlib/deflate stream
    if args.recompress >= 0:
        idat = [chunk[1] for chunk in png if chunk[0] == b'IDAT'][0]
        idat_size = len(idat)
        idat = zlib.compress(zlib.decompress(idat), level=args.recompress)
        print(f'Recompress: CL={args.recompress} ({idat_size} to {len(idat)} bytes)')
        png = before_idat.copy()
        png.append((b'IDAT', idat))
        png += after_idat
    # bloat zlib/deflate stream
    if args.bloat:
        idat = [chunk[1] for chunk in png if chunk[0] == b'IDAT'][0]
        idat_size = len(idat)
        idat = encode_noncompress(zlib.decompress(idat))
        print(f'Recompress: bloating x{len(idat)/idat_size:.2f} ({idat_size} to {len(idat)} bytes)')
        png = before_idat.copy()
        png.append((b'IDAT', idat))
        png += after_idat
    return png


# process IDAT data
def process_idat(png, args):
    idat_idx = [idx for idx, chunk in enumerate(png) if chunk[0] == b'IDAT']
    before_idat = [chunk for idx, chunk in enumerate(png) if idx < idat_idx[0]]
    after_idat = [chunk for idx, chunk in enumerate(png) if idat_idx[0] < idx and idx not in idat_idx]
    # merge IDAT chunks
    if args.merge_idat:
        idat_chunks = [chunk[1] for chunk in png if chunk[0] == b'IDAT']
        if len(idat_chunks) > 1 and args.merge_idat:
            idat = bytearray().join(idat_chunks)
            print(f'Merge: {len(idat_chunks)} IDAT chunks ({len(idat)} bytes)')
            png = before_idat.copy()
            png.append((b'IDAT', idat))
            png += after_idat
    # recompress IDAT data
    png = recompress_idat(png, args)
    # split IDAT chunk
    if args.split_idat > 0:
        idat_chunks = [chunk[1] for chunk in png if chunk[0] == b'IDAT']
        if len(idat_chunks) > 1:
            print(f'Split: Input PNG has {len(idat_chunks)} IDAT chunks.')
            sys.exit(1)
        idat = idat_chunks[0]
        idat_size = len(idat)
        nchunk, rem = math.ceil(idat_size / args.split_idat), idat_size % args.split_idat
        print(f'Split: {nchunk} IDAT chunks ({args.split_idat},{rem} bytes/chunk)')
        png = before_idat.copy()
        pos = 0
        while pos < idat_size:
            end_pos = min(pos + args.split_idat, idat_size)
            png.append((b'IDAT', idat[pos:end_pos]))
            pos = end_pos
        png += after_idat
    return png


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
        if chunk[0] in args.keep:
            png.append(chunk)
        if chunk[0] == b'IHDR':
            ihdr = parse_IHDR(chunk)
        elif chunk[0] == b'IEND':
            break
        if chunk[0] in parser:
            parser[chunk[0]](chunk, ihdr)
    # process IDAT chunks
    png = process_idat(png, args)
    # write PNG chunks
    for chunk in png:
        write_chunk(fout, *chunk)


def main(args):
    print(f'Input: {args.infile}')
    print(f'Output: {args.outfile}')
    # process PNG format
    print(f'FilterChunk: keep={args.keep}')
    print(f'ProcessIDAT: merge={args.merge_idat} split={args.split_idat}')
    print(f'Recompress: CL={args.recompress} bloat={args.bloat}')
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
    parser.add_argument('--merge-idat', type=int, choices=(0, 1), default=1,
                        help='merge into single IDAT chunk (default: %(default)s)')
    parser.add_argument('--split-idat', metavar='N', type=int, default=0,
                        help='split N bytes to multiple IDAT chunks (default: None)')
    parser.add_argument('--recompress', metavar='CL', type=int, default=-1,
                        help='recompress zlib/deflate with CL=[0, 9] (default: None)')
    parser.add_argument('--bloat', action='store_true',
                        help='<WTF> bloat deflate stream with non-compressed block')
    args = parser.parse_args()
    main(args)
