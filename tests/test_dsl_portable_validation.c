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
    me_portable_status rc = me_validate_portable_dsl(source, "0.1", inputs, count, dtype, &error);
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
    assert(me_validate_portable_dsl(source, "0.1", inputs, count, dtype, NULL) == expected);
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
          ME_FLOAT64, ME_PORTABLE_ERR_UNSUPPORTED);
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
    check("def k(x, y):\n    return x + y\n", reversed, 2, ME_FLOAT64, ME_PORTABLE_ERR_UNSUPPORTED);

    const char *unsupported[] = {
        "def k(x):\n    return sin(x)\n",
        "def k(x):\n    return cos(x)\n",
        "def k(x):\n    return int(x + 0.25)\n",
        "def k(x):\n    return float(float(x))\n",
        "def k(x):\n    return bool((x + 1) - x)\n",
        "def k(x):\n    return (x / 2) > 0\n",
        "def k(x):\n    return sum(x)\n",
        "def k(x):\n    if any(x):\n        return x\n    return x\n",
        "def k(x):\n    while x:\n        print(x)\n    return x\n",
        "def k(x):\n    return _i0 + x\n",
        "def k(x):\n    return _flat_idx + x\n",
        "def k(x):\n    return x.lower()\n",
        "def k(x):\n    return np.sin(x)\n",
        "def k(x):\n    return x[0]\n",
        "def k(x):\n    return 'hello'\n",
        "def k(x):\n    return callback(x)\n",
        "def k(x):\n    return x ** 2\n",
        "def k(x):\n    return x // 2\n",
        "# me:fp=fast\ndef k(x):\n    return x\n"
    };
    for (size_t i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); i++) {
        check(unsupported[i], x, 1, ME_FLOAT64, ME_PORTABLE_ERR_UNSUPPORTED);
    }
    check("def k(x):\n    return missing + x\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SOURCE);
    check("def k(x):\n    return rand\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SOURCE);
    check("def k(x):\n    return sin()\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_UNSUPPORTED);
    check("def k( :\n", x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SOURCE);
    check(NULL, x, 1, ME_FLOAT64, ME_PORTABLE_ERR_SOURCE);
    x[0].dtype = ME_INT64;
    check("def k(x):\n    return x + 0.5\n", x, 1, ME_INT64, ME_PORTABLE_ERR_UNSUPPORTED);
    check("def k(x):\n    return x == 9007199254740993.0\n", x, 1, ME_BOOL, ME_PORTABLE_ERR_UNSUPPORTED);
    check("def k(x):\n    return x == 1e3\n", x, 1, ME_BOOL, ME_PORTABLE_ERR_UNSUPPORTED);
    check("def k(x):\n    return float(x)\n", x, 1, ME_FLOAT32, ME_PORTABLE_ERR_UNSUPPORTED);
    check("def k(x):\n    return x / 2\n", x, 1, ME_INT64, ME_PORTABLE_ERR_UNSUPPORTED);
    check("def k(x):\n    return x + 9007199254740994\n", x, 1,
          ME_INT64, ME_PORTABLE_ERR_UNSUPPORTED);
    check("def k(x):\n    return x + 9007199254740992\n", x, 1,
          ME_INT64, ME_PORTABLE_SUCCESS);
    check("def k(x):\n    return x + 9007199254740993\n", x, 1,
          ME_INT64, ME_PORTABLE_ERR_UNSUPPORTED);
    check("def k(x):\n    return x - 9_007_199_254_740_993\n", x, 1,
          ME_INT64, ME_PORTABLE_ERR_UNSUPPORTED);
    check("def k(x):\n    return x + 0x20000000000001\n", x, 1,
          ME_INT64, ME_PORTABLE_ERR_UNSUPPORTED);
    x[0].dtype = ME_FLOAT32;
    check("def k(x):\n    return (x + 1.0) > x\n", x, 1, ME_BOOL, ME_PORTABLE_ERR_UNSUPPORTED);
    check("def k(x):\n    return (x + 1.0) - x\n", x, 1, ME_FLOAT32, ME_PORTABLE_SUCCESS);
    x[0].dtype = ME_INT64;
    me_portable_error error;
    assert(me_validate_portable_dsl("def k(x):\n    return x\n", "0.2", x, 1,
                                   ME_INT64, &error) == ME_PORTABLE_ERR_VERSION);
    assert(error.line == 0 && strstr(error.message, "0.1"));
    x[0].dtype = ME_FLOAT64;
    assert(me_validate_portable_dsl("def k(x):\n    return sum(x)\n", "0.1", x, 1,
                                   ME_FLOAT64, &error) == ME_PORTABLE_ERR_UNSUPPORTED);
    assert(error.line == 2 && error.column > 0 && strstr(error.message, "sum"));
    return 0;
}
