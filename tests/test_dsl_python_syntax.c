/* Parser regressions for Python-style documentation and simple-statement lists. */
#include <stdio.h>
#include "../src/dsl_parser.h"

int main(void) {
    const char *docs[] = {
        "'Single line; # text'",
        "\"Double quoted\"",
        "\"\"\"Multiple\nno indentation; # text\n    lines\"\"\"",
        "'''Multiple\n    lines'''",
        "r\"Raw \\n text\"",
        "u'Unicode text'",
        "'Escaped \\' quote'",
        "\"\"\"Embedded \"quotes\" and 'apostrophes'\"\"\"",
    };
    for (size_t i = 0; i < sizeof(docs) / sizeof(docs[0]); i++) {
        char source[1024];
        snprintf(source, sizeof(source),
                 "def k(x):\n    # before docs\n\n    %s; a = x + 1; return a; # end\n", docs[i]);
        me_dsl_error error;
        me_dsl_program *program = me_dsl_parse(source, &error);
        if (!program || program->block.nstmts != 2) {
            printf("Docstring case %zu failed: %s\n", i, error.message);
            me_dsl_program_free(program);
            return 1;
        }
        me_dsl_program_free(program);
    }
    const char *bad[] = {
        "def k(x):\n    'unterminated\n    return x\n",
        "def k(x):\n    '''unterminated\n    return x\n",
        "def k(x):\n    a = 1;; return a\n",
        "def k(x):\n    a = 1; if x > 0:\n        return a\n",
        "def k(x):\n    a = 1; for i in range(3):\n        a += i\n    return a\n",
        "def k(x):\n    a = 1; while x > 0:\n        a += 1\n    return a\n",
        "def k(x):\n    for i in range(3):; a = 1\n        a = 2\n    return a\n",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        me_dsl_error error;
        me_dsl_program *program = me_dsl_parse(bad[i], &error);
        if (program || error.line != 2) {
            printf("Invalid syntax case %zu was accepted or mislocated\n", i);
            me_dsl_program_free(program);
            return 1;
        }
    }
    me_dsl_error error;
    me_dsl_program *program = me_dsl_parse(
        "def k(x):\n    '''Line one\nLine two\n    end'''\n    a = 1;; return a\n", &error);
    if (program || error.line != 5 || error.column != 11) {
        printf("Error after multiline docstring was mislocated: %d:%d\n", error.line, error.column);
        me_dsl_program_free(program);
        return 1;
    }
    return 0;
}
