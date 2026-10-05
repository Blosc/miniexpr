#include <stdio.h>
#include <string.h>
#include "../src/dsl_parser.h"

int main(void) {
    const char *valid[][2] = {
        {"1_000", "1000"}, {"0_0", "00"}, {"0b1010", "10"},
        {"0B_10_10", "10"}, {"0o755", "493"}, {"0O_7_5_5", "493"},
        {"0xff", "255"}, {"0X_FF", "255"}, {"0xCA_FE", "51966"},
        {"1_000.2_5", "1000.25"}, {".1_25", ".125"}, {"1_0.", "10."},
        {"1e1_0", "1e10"}, {"1_2.5e-0_2", "12.5e-02"},
        {"0_1.0", "01.0"}, {"-0b1010", "-10"},
        {"0xffff_ffff_ffff_ffff", "18446744073709551615"},
    };
    for (size_t i = 0; i < sizeof(valid) / sizeof(valid[0]); i++) {
        char source[1024];
        snprintf(source, sizeof(source), "def k(x):\n    pass; return %s\n", valid[i][0]);
        me_dsl_error error;
        me_dsl_program *program = me_dsl_parse(source, &error);
        if (!program || program->block.nstmts != 1 ||
            strcmp(program->block.stmts[0]->as.return_stmt.expr->text, valid[i][1])) {
            printf("Literal %s failed: %s\n", valid[i][0], error.message);
            me_dsl_program_free(program);
            return 1;
        }
        me_dsl_program_free(program);
    }
    const char *invalid[] = {
        "1__0", "1_", "1_.0", "1._0", "1e_2", "1e+_2", "1e2_",
        "0x", "0x_", "0x__ff", "0xfg", "0b102", "0o8", "012", "0_1",
        "0x1p2", "0x1.2", "0x1_0000_0000_0000_0000",
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        char source[1024];
        snprintf(source, sizeof(source), "def k(x):\n    return %s\n", invalid[i]);
        me_dsl_error error;
        me_dsl_program *program = me_dsl_parse(source, &error);
        if (program || error.line != 2 || error.column != 12) {
            printf("Invalid literal %s accepted or mislocated: %d:%d\n",
                   invalid[i], error.line, error.column);
            me_dsl_program_free(program);
            return 1;
        }
        snprintf(source, sizeof(source), "def k(x):\n    a = %s; 3\n    return x\n", invalid[i]);
        program = me_dsl_parse(source, &error);
        if (program || error.line != 2 || error.column != 9) {
            printf("Invalid assignment literal %s accepted or mislocated: %d:%d\n",
                   invalid[i], error.line, error.column);
            me_dsl_program_free(program);
            return 1;
        }
    }
    /* Numeric-looking strings and identifier suffixes must not be rewritten. */
    me_dsl_error error;
    me_dsl_program *program = me_dsl_parse(
        "def k(x_1):\n    pass\n    print('0x_FF; 1_000 # text'); return x_1\n", &error);
    if (!program || strcmp(program->block.stmts[0]->as.print_stmt.call->text,
                           "print('0x_FF; 1_000 # text')")) {
        printf("Strings or identifiers were changed\n");
        me_dsl_program_free(program);
        return 1;
    }
    me_dsl_program_free(program);
    return 0;
}
