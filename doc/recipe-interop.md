# b2o v2 interoperability fixtures

`tests/interop/recipe-v2.c` is a JSON-free C peer for Python-Blosc2's structured
MessagePack recipes. It is a **test codec**, not an installed persistence API or
a general-purpose MessagePack parser. File/container resolution belongs to the
embedding host, not miniexpr.

Build target `recipe_v2_peer`, then set Python-Blosc2's
`BLOSC2_MENUDET_INTEROP` to its executable when running
`tests/test_menudet_interop.py`. Tests exchange real `b2o` payloads in both
directions, including inline nested recipes and MessagePack extension 43.
Python also places the C-authored payload in a Blosc2 carrier and reopens it.

The first interoperable fixture subset is explicitly:

- b2o Menudet version 2, recipe version 1, language 0.2;
- four float64 lanes, two logical groups, elementwise cardinality;
- zero-input generators with exact IEEE float64 captures and logical coordinates;
- annotated float64 inputs supplied by nested inline recipes or extension 43;
- unsafe final casting, inferred output, context rank zero or one.

The C peer fails closed for versions, geometry, capture types or cardinality
outside that subset. It does not resolve arbitrary filesystem, dictionary,
remote, or field references. Production hosts must validate the complete recipe
schema and reference closure before evaluation; the C peer is not that validator.
Existing v1 JSON reads remain separately qualified in standard Blosc2 builds.
No change to formula reductions or graph JSON is implied.
