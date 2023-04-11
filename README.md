# pngutil
PNG utilities tool.

This tool processes the input PNG file as follows:

- filter specific chunk types (`--keep`),
- merge multiple IDAT chunkst to single IDAT (`--merge-idat`),
- split IDAT chunk to specific IDAT chunk size (`--split-idat`),
- recompress Deflate stream in IDAT chunk (`--recompress`),
- bloat/explode filesize with *insanity* Deflate encoder (`--bloat`,`--explode`).


# Details
## PNG filter types
This tool *not* modify PNG filter types (None/Sub/Up/Average/Peath) of each scanlines.

## Insanity Deflate encoder
`--bloat` option generates Deflate stream that consists of ONE 'non-compressed block' per ONE source byte.

`--explode` option generates Deflate stream that consisit of ONE 'Dynamic Huffman codes block' per ONE source byte.Each Deflate block builds most inefficient Huffman code table to get worst (longest) code lengths.


# License
MIT License
