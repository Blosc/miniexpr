/* Keep test expressions native while crossing UCRT libm through safe wrappers. */
#ifndef ME_TEST_COMPLEX_COMPAT_H
#define ME_TEST_COMPLEX_COMPAT_H

#include "../src/me_complex.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(_MSC_VER) && defined(__clang__)
#undef complex
#define complex _Complex
#undef I
#define I (1.0fi)
#define creal me_creal
#define crealf me_crealf
#define cimag me_cimag
#define cimagf me_cimagf
#define conj me_conj
#define conjf me_conjf
#define cabs me_cabs
#define cexp me_cexp
#define clog me_clog
#define csqrt me_csqrt
#endif

#endif
