/* Draft profile membership must be checked without Python or kernel execution. */
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../src/miniexpr.h"

static void check(const char *source, const me_variable *inputs, int count,
                  me_dtype dtype, me_portable_status expected) {
    me_portable_error error;
    memset(&error, 0x7f, sizeof(error));
    me_portable_status rc = me_validate_portable_dsl(source, "1.0", inputs, count, dtype, &error);
    if (rc != expected) {
        fprintf(stderr, "expected status %d, got %d at %d:%d: %s\n%s",
                expected, rc, error.line, error.column, error.message, source);
    }
    assert(rc == expected);
    if (rc == ME_PORTABLE_SUCCESS) {
        assert(error.line == 0 && error.column == 0 && error.message[0] == '\0');
    } else {
        assert(error.message[0] != '\0');
    }
    assert(me_validate_portable_dsl(source, "1.0", inputs, count, dtype, NULL) == expected);
}

static void descriptor_validation_fixture(void) {
    me_portable_error error;
    me_portable_validation_descriptor descriptor = {
        .struct_size = sizeof(descriptor), .version = ME_PORTABLE_DSL_VALIDATION_DESCRIPTOR_VERSION,
        .cardinality = ME_PORTABLE_ELEMENTWISE};
    me_variable x = {.name = "x", .dtype = ME_FLOAT32, .type = ME_VARIABLE};
    const me_dtype types[] = {ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64,
        ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64, ME_FLOAT32, ME_FLOAT64};
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        x.dtype = types[i];
        assert(me_validate_portable_dsl("def k(x):\n    return x\n", "1.0", &x, 1,
                                       x.dtype, &error) == ME_PORTABLE_SUCCESS);
        assert(!error.line && !error.column && !error.message[0]);
    }
    x.dtype = ME_FLOAT32;
    /* Compiler-selection pragma cannot invoke CC/TCC during validation. Runtime
     * domain/loop failures are not executed or guessed from sample arrays. */
    assert(me_validate_portable_dsl("# me:compiler=cc\ndef k(x):\n    return sin(x) / 0\n", "1.0",
                                   &x, 1, ME_FLOAT32, &error) == ME_PORTABLE_SUCCESS);
    assert(me_validate_portable_dsl_ex("def k(x):\n    return sum(x)\n", "1.0", &x, 1,
        ME_FLOAT32, &descriptor, &error) == ME_PORTABLE_ERR_SIGNATURE);
    descriptor.cardinality = ME_PORTABLE_BLOCK_SCALAR;
    assert(me_validate_portable_dsl_ex("def k(x):\n    return sum(x)\n", "1.0", &x, 1,
        ME_FLOAT32, &descriptor, NULL) == ME_PORTABLE_SUCCESS);
    descriptor.cardinality = ME_PORTABLE_ELEMENTWISE;
    assert(me_validate_portable_dsl_ex("def k(x):\n    return x + _i1\n", "1.0", &x, 1,
        ME_FLOAT32, &descriptor, &error) == ME_PORTABLE_ERR_SOURCE);
    assert(strstr(error.message, "rank"));
    descriptor.ndim = 2;
    assert(me_validate_portable_dsl_ex("def k(x):\n    return x + _i1\n", "1.0", &x, 1,
        ME_FLOAT32, &descriptor, &error) == ME_PORTABLE_SUCCESS);
    descriptor.ndim = 0;
    const char *invalid[] = {"def k(x):\n    print(x)\n    return x\n",
        "def k(x):\n    return callback(x)\n", "def k(x):\n    return x & 1\n", "def k(x):\n    return ~x\n",
        "def k(x):\n    return x[0]\n", "# me:fp=fast\ndef k(x):\n    return x\n"};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        assert(me_validate_portable_dsl(invalid[i], "1.0", &x, 1,
                                       ME_FLOAT32, &error) == ME_PORTABLE_ERR_SOURCE);
        assert(error.message[0]);
    }
    assert(me_validate_portable_dsl("def k(x):\n    return (\n", "1.0", &x, 1,
                                   ME_FLOAT32, &error) == ME_PORTABLE_ERR_SOURCE);
    assert(error.line > 0 && error.column > 0);
    me_variable pair[] = {{.name = "x", .dtype = ME_INT64}, {.name = "y", .dtype = ME_UINT64}};
    assert(me_validate_portable_dsl("def k(x, y):\n    return x < y\n", "1.0", pair, 2,
                                   ME_BOOL, &error) == ME_PORTABLE_SUCCESS);
    assert(me_validate_portable_dsl("def k(x, y):\n    return x + y\n", "1.0", pair, 2,
                                   ME_FLOAT64, &error) == ME_PORTABLE_ERR_SOURCE);
    x.dtype = ME_STRING;
    x.itemsize = 16;
    descriptor.output_itemsize = 16;
    assert(me_validate_portable_dsl_ex("def k(x):\n    return upper(x)\n", "1.0", &x, 1,
        ME_STRING, &descriptor, &error) == ME_PORTABLE_SUCCESS);
    descriptor.output_itemsize = 12;
    assert(me_validate_portable_dsl_ex("def k(x):\n    return upper(x)\n", "1.0", &x, 1,
        ME_STRING, &descriptor, &error) == ME_PORTABLE_ERR_SIGNATURE);
    x.itemsize = 3;
    assert(me_validate_portable_dsl_ex("def k(x):\n    return x\n", "1.0", &x, 1,
        ME_STRING, &descriptor, &error) == ME_PORTABLE_ERR_UNSUPPORTED);
    x.dtype = ME_BYTES;
    descriptor.output_itemsize = 3;
    assert(me_validate_portable_dsl_ex("def k(x):\n    return x\n", "1.0", &x, 1,
        ME_BYTES, &descriptor, &error) == ME_PORTABLE_SUCCESS);
    x.address = &x;
    assert(me_validate_portable_dsl_ex("def k(x):\n    return x\n", "1.0", &x, 1,
        ME_BYTES, &descriptor, &error) == ME_PORTABLE_ERR_SIGNATURE);
    x.address = NULL;
    x.type = ME_FUNCTION1;
    assert(me_validate_portable_dsl_ex("def k(x):\n    return x\n", "1.0", &x, 1,
        ME_BYTES, &descriptor, &error) == ME_PORTABLE_ERR_SIGNATURE);
    x.type = ME_VARIABLE;
    descriptor.version++;
    assert(me_validate_portable_dsl_ex("def k(x):\n    return x\n", "1.0", &x, 1,
        ME_BYTES, &descriptor, &error) == ME_PORTABLE_ERR_SIGNATURE);
}

int main(void) {
    me_variable x[] = {{"x", ME_FLOAT64, NULL, ME_VARIABLE, NULL, 0}};
    const me_dtype dtypes[] = {ME_BOOL, ME_INT32, ME_INT64, ME_FLOAT32, ME_FLOAT64};
    for (size_t i = 0; i < sizeof(dtypes) / sizeof(dtypes[0]); i++) {
        x[0].dtype = dtypes[i];
        check("def k(x):\n    return x\n", x, 1, dtypes[i], ME_PORTABLE_SUCCESS);
    }
    x[0].dtype = ME_FLOAT64;
    check("# me:compiler=cc\ndef k(x):\n    return sin(x) + cos(x)\n", x, 1,
          ME_FLOAT64, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    if x > 0:\n        return x\n", x, 1,
          ME_FLOAT64, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    while 1:\n        pass\n    return x\n", x, 1,
          ME_FLOAT64, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    for i in range(0, 3, 0):\n        pass\n    return x\n", x, 1,
          ME_FLOAT64, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return x and (x > 0)\n", x, 1, ME_BOOL, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return x + 0x_FF + 1_000.0\n", x, 1,
          ME_FLOAT64, ME_PORTABLE_SUCCESS);
    check("def k():\n    return 1.0\n", NULL, 0, ME_FLOAT64, ME_PORTABLE_SUCCESS);

    me_variable reversed[] = {
        {"y", ME_FLOAT64, NULL, ME_VARIABLE, NULL, 0},
        {"x", ME_FLOAT64, NULL, ME_VARIABLE, NULL, 0}
    };
    check("def k(x, y):\n    return x + y\n", reversed, 2, ME_FLOAT64, ME_PORTABLE_SUCCESS);
    reversed[0].name = "x";
    check("def k(x, y):\n    return x + y\n", reversed, 2, ME_FLOAT64, ME_PORTABLE_ERR_SIGNATURE);
    check("def k(y):\n    return y\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SIGNATURE);
    check("def k(x, y):\n    return x\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SIGNATURE);
    check("def k(x):\n    return x\n", NULL, 1, ME_FLOAT64, ME_PORTABLE_ERR_SIGNATURE);

    x[0].type = ME_FUNCTION1;
    check("def k(x):\n    return x\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SIGNATURE);
    x[0].type = ME_VARIABLE;
    x[0].address = x;
    check("def k(x):\n    return x\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SIGNATURE);
    x[0].address = NULL;
    x[0].dtype = ME_COMPLEX128;
    check("def k(x):\n    return x\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_UNSUPPORTED);
    x[0].dtype = ME_FLOAT64;
    check("def k(x):\n    return x\n", x, 1, ME_AUTO, ME_PORTABLE_ERR_UNSUPPORTED);
    reversed[0].name = "y";
    reversed[0].dtype = ME_INT32;
    check("def k(x, y):\n    return x + y\n", reversed, 2, ME_FLOAT64, ME_PORTABLE_SUCCESS);

    const char *admitted[] = {
        "def k(x):\n    return sin(x)\n",
        "def k(x):\n    return cos(x)\n",
        "def k(x):\n    return int(x + 0.25)\n",
        "def k(x):\n    return float(float(x))\n",
        "def k(x):\n    return bool((x + 1) - x)\n",
        "def k(x):\n    return (x / 2) > 0\n",
        "def k(x):\n    return sum(x)\n",
        "def k(x):\n    if any(x):\n        return x\n    return x\n",
        "def k(x):\n    return x ** 2\n",
        "def k(x):\n    return x // 2\n"
    };
    for (size_t i = 0; i < sizeof(admitted) / sizeof(admitted[0]); i++) {
        check(admitted[i], x, 1, ME_FLOAT64, ME_PORTABLE_SUCCESS);
    }
    const char *unsupported[] = {
        "def k(x):\n    while x:\n        print(x)\n    return x\n",
        "def k(x):\n    return _i0 + x\n",
        "def k(x):\n    return _flat_idx + x\n",
        "def k(x):\n    return x.lower()\n",
        "def k(x):\n    return np.sin(x)\n",
        "def k(x):\n    return x[0]\n",
        "def k(x):\n    return 'hello'\n",
        "def k(x):\n    return callback(x)\n",
        "# me:fp=fast\ndef k(x):\n    return x\n"
    };
    for (size_t i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); i++) {
        check(unsupported[i], x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SOURCE);
    }
    check("def k(x):\n    return missing + x\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SOURCE);
    check("def k(x):\n    return rand\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SOURCE);
    check("def k(x):\n    return sin()\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SOURCE);
    check("def k( :\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SOURCE);
    check(NULL, x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SOURCE);
    x[0].dtype = ME_INT64;
    check("def k(x):\n    return x + 0.5\n", x, 1, ME_INT64, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return x == 9007199254740993.0\n", x, 1, ME_BOOL, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return x == 1e3\n", x, 1, ME_BOOL, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return float(x)\n", x, 1, ME_FLOAT32, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return x / 2\n", x, 1, ME_INT64, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return x + 9007199254740994\n", x, 1,
          ME_INT64, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return x + 9007199254740992\n", x, 1,
          ME_INT64, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return x + 9007199254740993\n", x, 1,
          ME_INT64, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return x - 9_007_199_254_740_993\n", x, 1,
          ME_INT64, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return x + 0x20000000000001\n", x, 1,
          ME_INT64, ME_PORTABLE_SUCCESS);
    x[0].dtype = ME_FLOAT32;
    check("def k(x):\n    return (x + 1.0) > x\n", x, 1, ME_BOOL, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return (x + 1.0) - x\n", x, 1, ME_FLOAT32, ME_PORTABLE_SUCCESS);
    x[0].dtype = ME_INT64;
    me_portable_error error;
    assert(me_validate_portable_dsl("def k(x):\n    return x\n", "0.2", x, 1,
                                   ME_INT64, &error) == ME_PORTABLE_ERR_VERSION);
    assert(error.line == 0 && strstr(error.message, "1.0"));
    x[0].dtype = ME_FLOAT64;
    assert(me_validate_portable_dsl("def k(x):\n    return sum(x)\n", "0.1", x, 1,
                                   ME_FLOAT64, &error) == ME_PORTABLE_ERR_VERSION);
    assert(error.line == 0 && strstr(error.message, "unsupported"));
    descriptor_validation_fixture();
    return 0;
}
