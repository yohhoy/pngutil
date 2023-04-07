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


# bit writer for Deflate(RFC1951)
class BitWriter():
    def __init__(self):
        self.data = bytearray()
        self.blen = 0
        self.bbuf = 0
    # write n-bits
    def bits(self, value, n):
        assert value <= (1 << n) - 1
        while 0 < n:
            m = min(n, 8 - self.blen)
            mb = (1 << m) - 1
            self.bbuf |= (value & mb) << self.blen
            self.blen += m
            value >>= m
            n -= m
            if self.blen == 8:
                self.data.append(self.bbuf)
                self.blen = 0
                self.bbuf = 0
        return self
    def flush(self):
        self.data.append(self.bbuf)
        self.blen = 0
        self.bbuf = 0
        return self.data


# Huffman encoder
class HuffmanEncoder():
    def __init__(self, lens):
        # huffman lengths to huffman codes (RFC1951, 3.2.2)
        def len2code(lens):
            # step1
            MAX_BITS = max(lens)
            bl_count = [0] * (MAX_BITS + 1)
            for l in lens:
                bl_count[l] += 1
            # step2
            code = 0
            bl_count[0] = 0
            next_code = [0] * (MAX_BITS + 1)
            for bits in range(1, MAX_BITS + 1):
                code = (code + bl_count[bits-1]) << 1
                next_code[bits] = code
            # step3
            codes = [0] * len(lens)
            for n, l in enumerate(lens):
                if l != 0:
                    codes[n] = next_code[l]
                    next_code[l] += 1
            return codes
        self.lens = lens
        self.codes = len2code(lens)
        # reverse bits in codes
        assert len([1 for n, c in zip(lens, self.codes) if c >= (1 << n)]) == 0
        def revbits(v, n):
            rv = 0
            for i in range(n):
                rv <<= 1
                rv |= (v >> i) & 1
            return rv
        self.rcodes = [revbits(v, n) for v, n in zip(self.codes, lens)]
    # encode symbol
    def encode(self, w, sym):
        w.bits(self.rcodes[sym], self.lens[sym])
    # debug: codes list
    def codes_str(self):
        return ', '.join([f'{c:0{l}b}' if l else '-' for c,l in zip(self.codes, self.lens)])


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


# encode zlib/deflate 'non-compressed block' per ONE byte
def encode_noncompress(data):
    stream = bytearray(b'\x78\x01')   
    for b in data[:-1]:
        stream += b'\x00\x01\x00\xfe\xff' # BFINAL=0 BTYPE=00
        stream.append(b)
    stream += b'\x01\x01\x00\xfe\xff' # BFINAL=1 BTYPE=00
    stream.append(data[-1])
    stream += struct.pack('>I', zlib.adler32(data))
    # SANITY CHECK
    assert data == zlib.decompress(stream)
    return stream


# encode zlib/deflate 'dynamic Huffman codes block' per ONE byte
def encode_dynamichuffman(data):
    # https://github.com/madler/zlib/blob/v1.2.13/inftrees.c#L130-L138
    def zlib_validate_lens(lens):
        MAX_BITS = max(lens)
        count = [0] * (MAX_BITS + 1)
        for n in lens:
            count[n] += 1
        left = 1
        for n in range(1, MAX_BITS + 1):
            left <<= 1
            left -= count[n]
            assert left >= 0, 'over-subscribed'
        assert left <= 0 or MAX_BITS == 0, 'incomplete set'
    stream = bytearray(b'\x78\x01')
    # https://github.com/madler/zlib/blob/v1.2.13/inflate.c#L946
    HLIT, HDIST, HCLEN = 286, 30, 19
    DEFLATE_CLEN_ORD = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
    CLEN = [0] * 3 + [4] * 16
    assert len(CLEN) == HCLEN
    clen = [CLEN[DEFLATE_CLEN_ORD.index(n)] for n in range(HCLEN)]
    zlib_validate_lens(clen)  # avoid 'invalid code lengths set'
    henc_clen = HuffmanEncoder(clen)
    print(f'  CLEN={clen}')
    print(f'  CLEN_CODES={henc_clen.codes_str()}')
    LIT_LENS = [15] + [8] * 255 + [15] + [9, 10, 11, 12, 13, 14] + [0] * 23
    assert len(LIT_LENS) == HLIT
    zlib_validate_lens(LIT_LENS)  # avoid 'invalid literal/lengths set'
    henc_lit = HuffmanEncoder(LIT_LENS)
    print(f'  LIT_LENS={LIT_LENS}')
    print(f'  LIT_CODES={henc_lit.codes_str()}')
    DIST_LENS = [0] * HDIST
    zlib_validate_lens(DIST_LENS)  # avoid 'invalid distances set'
    henc_dist = HuffmanEncoder(DIST_LENS)
    print(f'  DIST_LENS={DIST_LENS}')
    print(f'  DIST_CODES={henc_dist.codes_str()}')
    w = BitWriter()
    for n, b in enumerate(data):
        bfinal = 1 if (n == len(data) - 1) else 0
        w.bits(bfinal, 1)   # BFINAL
        w.bits(0b10, 2)     # BTYPE=10
        w.bits(HLIT - 257, 5)
        w.bits(HDIST - 1, 5)
        w.bits(HCLEN - 4, 4)
        for n in CLEN:
            w.bits(n, 3)
        for s in LIT_LENS:
            henc_clen.encode(w, s)
        for s in DIST_LENS:
            henc_clen.encode(w, s)
        henc_lit.encode(w, b)
        henc_lit.encode(w, 256) # end of blocks
    stream += w.flush()
    stream += struct.pack('>I', zlib.adler32(data))
    # SANITY CHECK
    assert data == zlib.decompress(stream)
    return stream


# recompress IDAT data
def recompress_idat(png, args):
    if args.recompress == -1 and not (args.bloat or args.explode):
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
    # custom zlib/deflate stream
    if args.bloat or args.explode:
        idat = [chunk[1] for chunk in png if chunk[0] == b'IDAT'][0]
        idat_size = len(idat)
        if args.explode:
            idat = encode_dynamichuffman(zlib.decompress(idat))
            print(f'Recompress: exploding x{len(idat)/idat_size:.2f} ({idat_size} to {len(idat)} bytes)')
        if args.bloat:
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
    print(f'Recompress: CL={args.recompress} bloat={args.bloat} explode={args.explode}')
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
    parser.add_argument('--explode', action='store_true',
                        help='<WTF> explode deflate stream with dynamic Huffman codes block')
    args = parser.parse_args()
    main(args)
